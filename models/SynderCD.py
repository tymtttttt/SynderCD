import types
import math
import pywt
import pywt.data
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .vmamba import SS2D
from scipy.io import savemat
from .CLIP.clip.clip import tokenize
from .CLIP.clip.clip import load

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import os
import cv2

def feature_vis(feats, name, output_dir='./feature_vis'):
    # 通道聚合（取均值）
    channel_agg = torch.mean(feats, dim=1, keepdim=True)  # [b,1,h,w]
    
    # 上采样到统一尺寸
    channel_agg = F.interpolate(channel_agg, size=(256,256), mode='bilinear')
    
    # 转换为numpy并归一化到[0,255]
    feat_np = channel_agg.squeeze().cpu().detach().numpy()  # [h,w]
    feat_norm = ((feat_np - np.min(feat_np)) / (np.max(feat_np) - np.min(feat_np) + 1e-8) * 255)
    
    # **关键修改：反转数值，使高激活区变红**
    feat_inverted = 255 - feat_norm.astype(np.uint8)  # 反转后高值→红，低值→蓝
    
    # 应用JET颜色映射
    heatmap = cv2.applyColorMap(feat_inverted, cv2.COLORMAP_JET)
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f"{output_dir}/{name}.png", heatmap)
    
################## Encoder Modules ####################
class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        return x, H, W
        
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.pos(x)
        x = self.act(x)
        x = self.fc2(x)

        return x

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters
 
def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)
        
def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x
    
class WEMamba(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1',ssm_ratio=2,forward_type="v05",):
        super(WEMamba, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.global_atten = SS2D(d_model=in_channels, dropout=0, d_state=16)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )

        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        
        x = x.permute(0, 2, 3, 1)
        x = self.global_atten(x)
        x = x.permute(0, 3, 1, 2)
        
        x = self.base_scale(x)
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class LoFiConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_weight_init=1):
        super().__init__()
        self.add_module('dwconv3x3',
                        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=in_channels,
                                  bias=False))
        self.add_module('bn1', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('dwconv1x1',
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=in_channels,
                                  bias=False))
        self.add_module('bn2', nn.BatchNorm2d(out_channels))

        # Initialize batch norm weights
        nn.init.constant_(self.bn1.weight, bn_weight_init)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, bn_weight_init)
        nn.init.constant_(self.bn2.bias, 0)

    @torch.no_grad()
    def fuse(self):
        # Fuse dwconv3x3 and bn1
        dwconv3x3, bn1, relu, dwconv1x1, bn2 = self._modules.values()

        w1 = bn1.weight / (bn1.running_var + bn1.eps) ** 0.5
        w1 = dwconv3x3.weight * w1[:, None, None, None]
        b1 = bn1.bias - bn1.running_mean * bn1.weight / (bn1.running_var + bn1.eps) ** 0.5

        fused_dwconv3x3 = nn.Conv2d(w1.size(1) * dwconv3x3.groups, w1.size(0), w1.shape[2:], stride=dwconv3x3.stride,
                                    padding=dwconv3x3.padding, dilation=dwconv3x3.dilation, groups=dwconv3x3.groups,
                                    device=dwconv3x3.weight.device)
        fused_dwconv3x3.weight.data.copy_(w1)
        fused_dwconv3x3.bias.data.copy_(b1)

        # Fuse dwconv1x1 and bn2
        w2 = bn2.weight / (bn2.running_var + bn2.eps) ** 0.5
        w2 = dwconv1x1.weight * w2[:, None, None, None]
        b2 = bn2.bias - bn2.running_mean * bn2.weight / (bn2.running_var + bn2.eps) ** 0.5

        fused_dwconv1x1 = nn.Conv2d(w2.size(1) * dwconv1x1.groups, w2.size(0), w2.shape[2:], stride=dwconv1x1.stride,
                                    padding=dwconv1x1.padding, dilation=dwconv1x1.dilation, groups=dwconv1x1.groups,
                                    device=dwconv1x1.weight.device)
        fused_dwconv1x1.weight.data.copy_(w2)
        fused_dwconv1x1.bias.data.copy_(b2)

        # Create a new sequential model with fused layers
        fused_model = nn.Sequential(fused_dwconv3x3, relu, fused_dwconv1x1)
        return fused_model

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m
        
def nearest_multiple_of_16(n):
    if n % 16 == 0:
        return n
    else:
        lower_multiple = (n // 16) * 16
        upper_multiple = lower_multiple + 16

        if (n - lower_multiple) < (upper_multiple - n):
            return lower_multiple
        else:
            return upper_multiple
            
class TSCModule(torch.nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25,
                 kernels=3, ssm_ratio=1, forward_type="v052d",):
        super().__init__()
        self.dim = dim
        self.global_channels = nearest_multiple_of_16(int(global_ratio * dim))
        if self.global_channels + int(local_ratio * dim) > dim:
            self.local_channels = dim - self.global_channels
        else:
            self.local_channels = int(local_ratio * dim)
        self.identity_channels = self.dim - self.global_channels - self.local_channels
        if self.local_channels != 0:
            self.local_op = LoFiConv(self.local_channels, self.local_channels, kernels)
        else:
            self.local_op = nn.Identity()
        if self.global_channels != 0:
            self.global_op = WEMamba(self.global_channels, self.global_channels, kernels, wt_levels=1, ssm_ratio=ssm_ratio, forward_type=forward_type,)
        else:
            self.global_op = nn.Identity()

        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            dim, dim, bn_weight_init=0,))

    def forward(self, x):  # x (B,32,64,64)
        x1, x2, x3 = torch.split(x, [self.global_channels, self.local_channels, self.identity_channels], dim=1)# x1 (B,16,64,64) x2 (B,8,64,64) x3 (B,8,64,64)
        x2 = self.local_op(x2)
        x1 = self.global_op(x1)
        x = self.proj(torch.cat([x1, x2, x3], dim=1))+x
        return x
        
class EncoderBlock(nn.Module):
    def __init__(self, dim, drop_path=0.1, mlp_ratio=4, heads=4):
        super().__init__()

        self.layer_norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio)
        self.att = TSCModule(dim)
        self.drop_path = DropPath(drop_path)
    def forward(self, x, H, W):
        B, C, H, W = x.shape
        inp_copy = x
              
        x = self.layer_norm1(inp_copy)
        x = self.att(x)
        x = self.drop_path(x)
        out = x + inp_copy

        x = self.layer_norm2(out)
        x = self.mlp(x)
        x = self.drop_path(x)
        out = out + x
        return out

class PatchMerging(nn.Module):
    """
    Modified patch merging layer that works as down-sampling
    """

    def __init__(self, dim, out_dim, norm=nn.BatchNorm2d, proj_type='depthwise'):
        super().__init__()
        self.dim = dim
        if proj_type == 'linear':
            self.reduction = nn.Conv2d(4*dim, out_dim, kernel_size=1, bias=False)
        else:
            self.reduction = DepthwiseSeparableConv(4*dim, out_dim)

        self.norm = norm(4*dim)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]

        x = torch.cat([x0, x1, x2, x3], 1) # B, 4C, H, W

        x = self.norm(x)
        x = self.reduction(x)
        _, _, H, W = x.shape
        return x, H, W

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias
        )
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out
        
################## Encoder #########################
#Transormer Ecoder with x4, x8, x16, x32 scales
class Encoder(nn.Module):
    def __init__(self, patch_size=3, in_chans=3, embed_dims=[32, 64, 128, 256],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], drop_path_rate=0., heads=[4, 4, 4, 4],
                 depths=[3, 3, 4, 3]):
        super().__init__()
        self.depths         = depths
        self.embed_dims     = embed_dims
        
        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        
        self.block1 = nn.ModuleList()
        for i in range(depths[0]):
            self.block1.append(EncoderBlock(dim=embed_dims[0], mlp_ratio=mlp_ratios[0]))

        self.block2 = nn.ModuleList()
        for i in range(depths[1]):
            self.block2.append(EncoderBlock(dim=embed_dims[1], mlp_ratio=mlp_ratios[1]))
        
        self.block3 = nn.ModuleList()
        for i in range(depths[2]):
            self.block3.append(EncoderBlock(dim=embed_dims[2], mlp_ratio=mlp_ratios[2]))   

        self.block4 = nn.ModuleList()
        for i in range(depths[3]):
            self.block4.append(EncoderBlock(dim=embed_dims[3], mlp_ratio=mlp_ratios[3]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x):

        B = x.shape[0]
        outs = []
    
        # stage 1
        x1, H1, W1 = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x1 = blk(x1, H1, W1)
        outs.append(x1)
        
        # stage 2
        x1, H1, W1 = self.patch_embed2(x1)
        for i, blk in enumerate(self.block2):
            x1 = blk(x1, H1, W1)
        outs.append(x1)
        # stage 3
        x1, H1, W1 = self.patch_embed3(x1)
        for i, blk in enumerate(self.block3):
            x1 = blk(x1, H1, W1)
        outs.append(x1)

        # stage 4
        x1, H1, W1 = self.patch_embed4(x1)
        for i, blk in enumerate(self.block4):
            x1 = blk(x1, H1, W1)
        outs.append(x1)
        
        
        return outs

###########  deocde Modules ###############
class LinearProj(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(input_dim, embed_dim, 3, stride=1, padding=1)
        self.BN = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.proj(x)
        x = self.BN(x)
        x = self.relu(x)
        return x
        
class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Fusion_Block(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(Fusion_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels*2, out_channels, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inp_feats):
        x1, x2 = inp_feats[0], inp_feats[1]
        x = self.relu(self.bn(self.conv(torch.cat([x1,x2], dim=1))))

        return x

def generate_prompt_from_x(x):
    """
    根据 T1 和 T2 的差分图 x 生成 CLIP 兼容的文本提示
    """
    B, C, H, W = x.shape
    
    # 计算变化区域的均值（整体变化程度）
    change_intensity = torch.mean(torch.abs(x), dim=[1, 2, 3])  
    threshold = 0.1 * x.std()
    change_ratio = torch.mean((torch.abs(x) > threshold).float(), dim=[1, 2, 3])

    prompts = []
    for i in range(B):
        # 变化占比描述
        if change_ratio[i] < 0.05:
            change_desc = "无明显变化"
        elif change_ratio[i] < 0.2:
            change_desc = "轻微变化"
        elif change_ratio[i] < 0.5:
            change_desc = "中等变化"
        else:
            change_desc = "大范围变化"

        # 变化强度描述
        if change_intensity[i] < 10:
            intensity_desc = "细微"
        elif change_intensity[i] < 50:
            intensity_desc = "中等"
        else:
            intensity_desc = "剧烈"

        # 生成简短的 Prompt
        prompt = f"该遥感影像显示 {change_desc}，变化强度 {intensity_desc}。"
        prompts.append(prompt)
    
    return prompts

class CLIPTextEncoder:
    def __init__(self, device, model_name="ViT-B/32"):
        self.device = device
        self.model, _ = load(model_name, device=device)
        self.model.eval()

    def encode(self, prompts, batch_size, pooling_method='mean'):
        tokens = tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_feats = self.model.encode_text(tokens)  # [N, 512]
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        # 聚合多个 prompt，支持选择池化方法
        if pooling_method == 'mean':
            text_feat = text_feats.mean(dim=0, keepdim=True)  # [1, 512]
        elif pooling_method == 'max':
            text_feat, _ = text_feats.max(dim=0, keepdim=True)  # [1, 512]
        else:
            raise ValueError("Unsupported pooling method")
        
        text_feat = text_feat.repeat(batch_size, 1)  # [B, 512]
        text_feat.requires_grad_()
        return text_feat

'''
class DecodeWithTextCrossAttention(nn.Module):
    def __init__(self, in_channel, out_channel, text_dim=512, num_heads=4):
        super(DecodeWithTextCrossAttention, self).__init__()
        self.text_encoder = CLIPTextEncoder(device=0)
        self.num_heads = num_heads
        self.head_dim = out_channel // num_heads
        assert out_channel % num_heads == 0, "out_channel must be divisible by num_heads"

        # 将文本嵌入映射为 Query
        self.text_query = nn.Linear(text_dim, out_channel)

        # 将图像特征映射为 Key 和 Value
        self.key_conv = nn.Conv2d(out_channel, out_channel, kernel_size=1)
        self.value_conv = nn.Conv2d(out_channel, out_channel, kernel_size=1)

        self.out_proj = nn.Linear(out_channel, out_channel)
        
        self.conv3 = nn.Conv2d(out_channel, out_channel,kernel_size=1)
    def forward(self, left, x):
        B = left.size(0)

        # 通过 x（T2 - T1 差分图）生成自适应文本提示
        prompts = generate_prompt_from_x(x)
        # 通过 CLIP 文本编码器获取文本特征
        text_embedding = self.text_encoder.encode(prompts, batch_size=B)  # [B, 512]
        
        key = self.key_conv(left)
        value = self.value_conv(left)

        key = key.view(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)
        value = value.view(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)

        query = self.text_query(text_embedding.to(self.text_query.weight.dtype)).view(B, self.num_heads, 1, self.head_dim)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.permute(0, 1, 3, 2).reshape(B, -1)    
        attn_output = self.out_proj(attn_output).view(B, -1, 1, 1)
        out = left * attn_output + left

        return out
'''       

class VALAttention(nn.Module):
    def __init__(self, in_channel, out_channel, text_dim=512, num_heads=4):
        super(DecodeWithTextCrossAttention, self).__init__()
        self.text_encoder = CLIPTextEncoder(device=1)
        self.num_heads = num_heads
        self.head_dim = out_channel // num_heads
        assert out_channel % num_heads == 0, "out_channel must be divisible by num_heads"

        # 文本注意力分支
        self.text_query = nn.Linear(text_dim, out_channel)
        self.key_conv = nn.Conv2d(out_channel, out_channel, kernel_size=1)
        self.value_conv = nn.Conv2d(out_channel, out_channel, kernel_size=1)
        self.out_proj = nn.Linear(out_channel, out_channel)
        
        # 局部卷积分支
        self.local_conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, groups=out_channel),  # 深度可分离卷积
            nn.Conv2d(out_channel, out_channel, kernel_size=1),  # 点卷积
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        
        # 融合门控机制
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//2, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, left, x):
        B = left.size(0)
        
        # 文本注意力分支
        prompts = generate_prompt_from_x(x)
        text_embedding = self.text_encoder.encode(prompts, batch_size=B)
        
        key = self.key_conv(left)
        value = self.value_conv(left)
        
        key = key.view(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)
        print(key.shape)
        value = value.view(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)
        print(value.shape)
        
        query = self.text_query(text_embedding.to(self.text_query.weight.dtype)).view(B, self.num_heads, 1, self.head_dim)
        print(query.shape)
        exit()
        
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.permute(0, 1, 3, 2).reshape(B, -1)    
        text_feat = self.out_proj(attn_output).view(B, -1, 1, 1)
        text_out = left * text_feat
        
        # 局部卷积分支
        local_feat = self.local_conv(left)
        
        # 自适应融合
        concat_feat = torch.cat([text_out, local_feat], dim=1)
        gates = self.fusion_gate(concat_feat)
        fused_out = gates[:, 0:1] * text_out + gates[:, 1:2] * local_feat
        
        # 残差连接
        out = fused_out + left
        
        return out
        
class VALAttention1(nn.Module):
    def __init__(self, in_channel, out_channel, text_dim=512, num_heads=4):
        super(DecodeWithTextCrossAttention1, self).__init__()
        self.conv1 = BasicConv2d(in_channel, out_channel, 3, padding=1)
        self.conv2 = BasicConv2d(out_channel, out_channel, 3, padding=1)
        self.text_encoder = CLIPTextEncoder(device=1)
        self.num_heads = num_heads
        self.head_dim = out_channel // num_heads
        assert out_channel % num_heads == 0, "out_channel must be divisible by num_heads"

        # 将文本嵌入映射为 Query
        self.text_query = nn.Linear(text_dim, out_channel)

        # 将图像特征映射为 Key 和 Value
        self.key_conv = nn.Conv2d(out_channel, out_channel, kernel_size=1)
        self.value_conv = nn.Conv2d(out_channel, out_channel, kernel_size=1)

        self.out_proj = nn.Linear(out_channel, out_channel)
        
                # 局部卷积分支
        self.local_conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, groups=out_channel),  # 深度可分离卷积
            nn.Conv2d(out_channel, out_channel, kernel_size=1),  # 点卷积
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        
        # 融合门控机制
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//2, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, left, down, x):
        B = left.size(0)

        down_feat = self.conv1(down)
        down_up = F.interpolate(down_feat, size=left.size()[2:], mode='bilinear', align_corners=False)
        fused = self.conv2(left + down_up)

        # 通过 x（T2 - T1 差分图）生成自适应文本提示
        prompts = generate_prompt_from_x(x)
        # 通过 CLIP 文本编码器获取文本特征
        text_embedding = self.text_encoder.encode(prompts, batch_size=B)  # [B, 512]
        
        key = self.key_conv(fused)
        value = self.value_conv(fused)

        key = key.view(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)
        value = value.view(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)

        query = self.text_query(text_embedding.to(self.text_query.weight.dtype)).view(B, self.num_heads, 1, self.head_dim)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.permute(0, 1, 3, 2).reshape(B, -1)    

        attn_output = self.out_proj(attn_output).view(B, -1, 1, 1)
        text_out = fused * attn_output 
        
        local_feat = self.local_conv(fused)
        
        # 自适应融合
        concat_feat = torch.cat([text_out, local_feat], dim=1)
        gates = self.fusion_gate(concat_feat)
        fused_out = gates[:, 0:1] * text_out + gates[:, 1:2] * local_feat
        
        # 残差连接
        out = fused_out + fused
        
        return out
        
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x
                
class Decoder(nn.Module):
    def __init__(self, in_channels = [64, 96, 128, 256],output_nc=2):
        super(Decoder, self).__init__()
        
        self.in_channels     = in_channels
        self.output_nc       = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels       
        
        self.decode1 = VALAttention(c4_in_channels,c4_in_channels)
        self.decode2 = VALAttention1(c4_in_channels,c4_in_channels)
        self.decode3 = VALAttention1(c4_in_channels,c4_in_channels)
        self.decode4 = VALAttention1(c4_in_channels,c4_in_channels)

        self.diffc1 = Fusion_Block(c1_in_channels,c4_in_channels)
        self.diffc2 = Fusion_Block(c2_in_channels,c4_in_channels)
        self.diffc3 = Fusion_Block(c3_in_channels,c4_in_channels)
        self.diffc4 = Fusion_Block(c4_in_channels,c4_in_channels)
        
        self.final = nn.Sequential(
            Conv(256, 32, 3, bn=True, relu=True),
            Conv(32, output_nc, 3, bn=False, relu=False)
            )
        self.fina2 = nn.Sequential(
            Conv(256, 32, 3, bn=True, relu=True),
            Conv(32, output_nc, 3, bn=False, relu=False)
            )
        self.fina3 = nn.Sequential(
            Conv(256, 32, 3, bn=True, relu=True),
            Conv(32, output_nc, 3, bn=False, relu=False)
            )
        self.fina4 = nn.Sequential(
            Conv(256, 32, 3, bn=True, relu=True),
            Conv(32, output_nc, 3, bn=False, relu=False)
            )
        
    def forward(self, inputs1, inputs2):
        #img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = inputs1   
        c1_2, c2_2, c3_2, c4_2 = inputs2 
        feature_vis(c1_1, 'x1_1')
        feature_vis(c2_1, 'x1_2')
        feature_vis(c3_1, 'x1_3')
        feature_vis(c4_1, 'x1_4')
        feature_vis(c1_2, 'x2_1')
        feature_vis(c2_2, 'x2_2')
        feature_vis(c3_2, 'x2_3')
        feature_vis(c4_2, 'x2_4')
        #x = self.conv(torch.abs(x1-x2))
        # Stage 4: x1/32 scale
        c4 = self.diffc4([c4_1, c4_2])
        feature_vis(c4, 'f4')
        c3 = self.diffc3([c3_1, c3_2])
        feature_vis(c3, 'f3')
        c2 = self.diffc2([c2_1, c2_2])
        feature_vis(c2, 'f2')
        c1 = self.diffc1([c1_1, c1_2])
        feature_vis(c1, 'f1')
        
        x = torch.abs(c4_1-c4_2)
        
        out1 = self.decode1(c4,x)
        out2 = self.decode2(c3,out1,x)
        out3 = self.decode3(c2,out2,x)
        out4 = self.decode4(c1,out3,x)
        
        #Final prediction
        outputs1 = self.final(F.interpolate(out4, scale_factor=(4, 4), mode='bilinear'))
        feature_vis(outputs1, 'outputs1')
        outputs2 = self.fina2(F.interpolate(out3, scale_factor=(8, 8), mode='bilinear'))
        feature_vis(outputs2, 'outputs2')
        outputs3 = self.fina3(F.interpolate(out2, scale_factor=(16, 16), mode='bilinear'))
        feature_vis(outputs3, 'outputs3')
        outputs4 = self.fina4(F.interpolate(out1, scale_factor=(32, 32), mode='bilinear'))
        feature_vis(outputs4, 'outputs4')

        return outputs1,outputs2,outputs3,outputs4

# SynderCD
class SynderCD(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, depths=[4, 4, 7, 4], heads=[4, 4, 4, 4],
                 enc_channels=[64, 96, 128,256]):
        super(SynderCD, self).__init__()

        self.embed_dims = enc_channels
        self.depths     = depths

        # shared encoder
        self.enc = Encoder(patch_size=5, in_chans=input_nc, embed_dims=self.embed_dims,
                                         heads=heads, mlp_ratios=[4, 4, 4, 4], depths=self.depths)
        
        # decoder
        self.dec = Decoder(in_channels=self.embed_dims,  output_nc=output_nc)

    def forward(self, x1, x2):

        fx1, fx2 = [self.enc(x1), self.enc(x2)]
        feature_vis(fx1[0], 'x1_1')
        feature_vis(fx1[1], 'x1_2')
        feature_vis(fx1[2], 'x1_3')
        feature_vis(fx1[3], 'x1_4')
        feature_vis(fx2[0], 'x2_1')
        feature_vis(fx2[1], 'x2_2')
        feature_vis(fx2[2], 'x2_3')
        feature_vis(fx2[3], 'x2_4')
        
        change_map = self.dec(fx1, fx2)

        return change_map
