"""
PINet_cpx_cl 全局配置文件。

用法:
  from src.config import *
  # 可按需覆盖: BATCH_SIZE = 8
"""

import os
import numpy as np
import torch
from torchvision import transforms

# ========================================
#  路径
# ========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

REAL_DATA_DIR = os.path.join(PROJECT_ROOT, "real_data")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
SAVE_DIR = os.path.join(PROJECT_ROOT, "save_dir")

# 训练输入 / 输出目录（notebook 中按需覆盖）
DATA_DIR = os.path.join(DATASETS_DIR, "synth_dataset_v1")
OUTPUT_DIR = os.path.join(SAVE_DIR, "model_synth_dataset_v1")

# ========================================
#  物理参数
# ========================================
WAVELENGTH = 532e-9            # 光波长 (m)，绿光 532nm
PIXEL_SIZE = 1.67e-6           # 像素尺寸 (m)，New Real Data
IMAGE_SIZE = 256               # 图像空间尺寸 (pixels)
Nx, Ny = IMAGE_SIZE, IMAGE_SIZE

# 物理导出量
lambd = np.array([WAVELENGTH])
k = 2 * np.pi / lambd
dx = PIXEL_SIZE

# 坐标栅格 (用于 mydataset.py 中计算 TF)
Lx = Nx * dx
Ly = Ny * dx
fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / Lx, Nx)
fy = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / Ly, Ny)
FX, FY = np.meshgrid(fy, fx)

# ========================================
#  设备 & 图像变换
# ========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ========================================
#  模型参数
# ========================================
MODEL_TYPE = "v6"               # v6 / v7 / CICDNet
FOLD_ITERS = 4                  # PINet 物理迭代次数
CBAM_RATIO = 8                  # CBAM 注意力通道缩减比
ALPHA = 0.5                     # v7 的去噪-物理投影融合权重

# ========================================
#  训练参数
# ========================================
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 200
LR_GAMMA = 0.98                 # 学习率衰减因子 (ExponentialLR)
NUM_WORKERS = 4

NUM_TRAIN_SAMPLES = 1200
NUM_VAL_SAMPLES = 200
NUM_TEST_SAMPLES = 200

# 传播距离采样范围 (mm)，随机生成各 batch 的 TF
Z_LOW = 1.5
Z_HIGH = 3.0

# ========================================
#  损失权重 (ComplexLoss_amp_phs)
# ========================================
WEIGHT_AMP = 1.0
WEIGHT_PHS = 1.0
WEIGHT_DIFF = 1.0
