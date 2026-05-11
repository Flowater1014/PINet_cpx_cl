import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.config import WAVELENGTH, PIXEL_SIZE, IMAGE_SIZE, DEVICE

# ====== 路径 ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 参考复振幅数据
REF_MAT_PATH = os.path.join(BASE_DIR, "..", "real_data", "JZX1_type1_new.mat")
REF_KEY = "label"                          # complex128, shape (1074, 1919)

# 自然图像
NATURAL_IMG_DIR = os.path.join(BASE_DIR, "..", "datasets", "pinet_1600")
NATURAL_IMG_NUM = 1600                     # 1_256.jpg ~ 1600_256.jpg

# ====== 裁剪参数（从参考数据裁剪 UNet 训练/验证/测试集）======
PATCH_SIZE = 256
MARGIN = 100                               # 裁剪时避开边缘像素
N_TRAIN = 1000
N_VAL = 200
N_TEST = 200
CROP_SEED = 0

# ====== 训练超参 ======
BATCH_SIZE = 8
EPOCHS = 200
LR = 1e-3
NUM_WORKERS = 4

# ====== 生成数据集切分 ======
N_TRAIN_GEN = 1200
N_VAL_GEN = 200
N_TEST_GEN = 200

# ====== 输出路径 ======
UNET_DATA_DIR = os.path.join(BASE_DIR, "unet_dataset_JZX1")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, "..", "datasets", "synth_dataset_v1")
