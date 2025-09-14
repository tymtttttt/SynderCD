import torch
from thop import profile
from models.SynderCD import SynderCD

device = torch.device("cuda:1")  # 指定使用第1号GPU（从0开始）

# 定义模型并放到GPU1上
model = ELGCNet().to(device)
model.eval()

# 构造输入数据并放到GPU1上
input1 = torch.randn(1, 3, 256, 256).to(device)
input2 = torch.randn(1, 3, 256, 256).to(device)

# 计算 FLOPs 和 参数量
flops, params = profile(model, (input1, input2))
print('FLOPs: %.4f GFLOPs' % (flops / 1e9))
print('Parameters: %.4f M' % (params / 1e6))

# 测量推理时间（平均多次以更稳定）
with torch.no_grad():
    repetitions = 100
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    # 预热
    for _ in range(10):
        _ = model(input1, input2)

    # 正式计时
    for _ in range(repetitions):
        starter.record()
        _ = model(input1, input2)
        ender.record()
        torch.cuda.synchronize(device)  # 等待GPU1完成任务
        curr_time = starter.elapsed_time(ender)  # 毫秒
        timings.append(curr_time)

    avg_inference_time = sum(timings) / repetitions
    print('Average inference time on GPU 1: %.4f ms' % avg_inference_time)