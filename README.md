<div align="center">
<h1> SynerCD:Synergistic Tri-branch and Vision-Language Coupling for Remote Sensing Change Detection </h1>
</div>

## 🎈 News

- [2024.12.13] Training and inference code released

## 🚀 Introduction


The challenges: 
(a) In encoding phase, how to effectively integrate spatial- and frequency-domain information while maintaining boundary integrity and enabling global-local modeling; 
(b) In reconstruction phase, how to efficiently aggregate multi-scale features while preserving high resolution.

## 📻 Overview

<div align="center">
<img width="800" alt="image" src="assert/SynderCD.PNG?raw=true">
</div>


Illustrates the overall architecture of SynderCD, which mainly consists of Tri-branch Synergistic Coupling Module. (a) The proposed Vision-Aware Language-guided Attention.

## 🎮 Getting Started

### 1. Install Environment

```
conda create -n ADCNet python=3.8
conda activate ADCNet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
```

### 2. Prepare Datasets

- Download datasets: LEVIR-CD from this [link](https://justchenhao.github.io/LEVIR), SYSU-CD from this [link](https://gitee.com/fuzhou-university-wq_0/SYSU-CD), and CDD-CD from this [link](https://aistudio.baidu.com/aistudio/datasetdetail/89523).

### 3. Train the SynerCD

```
python train.py --datasets LEVIR-CD
concrete information see ./SynerCD/train.py, please
```

### 3. Test the SynerCD

```
python test.py --datasets LEVIR-CD
testing results are saved to ./vis folder
concrete information see ./SynerCD/test.py, please
```


## 🖼️ Visualization

<div align="center">
<img width="800" alt="image" src="assert/Visualization.PNG?raw=true">
</div>



## 🎫 License

The content of this project itself is licensed under [LICENSE](LICENSE).
