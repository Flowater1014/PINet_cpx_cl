# PINet_cpx_cl

基于物理信息神经网络（PINet）的复振幅重建项目，用于无透镜成像 / 同轴全息相位恢复。

## 项目结构

```
PINet_cpx_cl/
├── src/                          # 核心源码
│   ├── config.py                 # 全局配置（物理参数、训练超参、路径）
│   ├── model.py                  # PINet_v6/v7/CICDNet + 复/实数 UNet
│   ├── CI_CDNet.py               # CI_CDNet 网络 + 权重初始化
│   ├── basicblock.py             # CI_CDNet 基础模块（复数卷积分量、残差块）
│   ├── complexFunctions.py       # 复数运算底层函数
│   ├── complexLayers.py          # 复数运算 nn.Module 层
│   ├── mydataset.py              # Dataset 类 + collate_fn
│   ├── myloss.py                 # 损失函数（幅相、实虚部）
│   └── utilities.py              # 物理传播、PSNR/SSIM、可视化工具
├── dataset_construction/          # 数据集构造流水线
│   ├── config.py                 # 构造流水线配置
│   ├── model.py                  # 实数 U-Net（振幅→相位）
│   ├── crop_dataset.py           # 从参考复振幅裁剪训练 patch
│   ├── train_unet.py             # 训练振幅→相位预测 U-Net
│   ├── generate_dataset.py       # 直方图匹配 + 预测相位 → 合成数据集
│   ├── utils.py                  # 数据 IO、直方图匹配
│   └── visualize.py / visualize.ipynb  # 可视化检查
├── notebooks/                    # 训练和测试脚本
│   ├── pretrain_new.ipynb        # PINet 预训练（几何增强）
│   └── test_models.ipynb         # 模型测试（真实数据推理）
├── datasets/                     # 数据集（不在版本控制中）
├── real_data/                    # 真实实验数据（.mat，不在版本控制中）
├── save_dir/                     # 训练好的模型权重和日志（不在版本控制中）
└── requirements.txt
```

## 快速开始

```bash
git clone git@github.com:Flowater1014/PINet_cpx_cl.git
cd PINet_cpx_cl
pip install -r requirements.txt
```

## 训练

1. 准备数据：将数据集放入 `datasets/`，真实数据放入 `real_data/`
2. 修改 `src/config.py` 中的参数（数据路径、训练超参、传播距离等）
3. 运行 `notebooks/pretrain_new.ipynb` 进行 PINet 预训练

```python
from src.config import *
# 可按需覆盖
BATCH_SIZE = 8
Z_LOW, Z_HIGH = 1.5, 3.0
```

## 测试

运行 `notebooks/test_models.ipynb`，修改 `data_name` 切换测试数据集。

## 合成数据集

当真实复振幅标注数据不足时，可使用 `dataset_construction/` 流水线从自然图像构造合成数据集：

```bash
cd dataset_construction
python crop_dataset.py       # ① 从参考复振幅裁剪训练 patch
python train_unet.py         # ② 训练振幅→相位预测 U-Net
python generate_dataset.py   # ③ 直方图匹配 + 预测相位 → 合成数据集
```

## 配置系统

`src/config.py` 集中管理所有参数，notebook 中 `from src.config import *` 即可使用，局部覆盖只改对应变量。
