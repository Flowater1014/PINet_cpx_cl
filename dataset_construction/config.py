import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.config import WAVELENGTH, PIXEL_SIZE, IMAGE_SIZE, DEVICE

# ====== 路径 ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 参考复振幅数据
REF_MAT_PATH = os.path.join(BASE_DIR, "..", "real_data", "Eout_beewings_unwrapped.mat")
REF_KEY = "P_HSSA_unwrap"                  # complex64, shape (4001, 5423)

# 自然图像
NATURAL_IMG_DIR = os.path.join(BASE_DIR, "..", "datasets", "pinet_1600")
NATURAL_IMG_NUM = 1600                     # 1_256.jpg ~ 1600_256.jpg

# ====== 裁剪参数（从参考数据裁剪 UNet 训练/验证/测试集）======
PATCH_SIZE = 256
MARGIN_TOP = 300
MARGIN_BOTTOM = 300
MARGIN_LEFT = 200
MARGIN_RIGHT = 200
N_H = 25                                    # x 方向网格数 (W=5423)
N_V = 20                                    # y 方向网格数 (H=4001)

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
UNET_DATA_DIR = os.path.join(BASE_DIR, "unet_dataset_Beewings")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
CKPT_DIR_4LAYER = os.path.join(BASE_DIR, "checkpoints_Beewings")
OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, "..", "datasets", "pinet_dataset_compared_7")
