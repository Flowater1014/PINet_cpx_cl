# PINet_cpx_cl

基于**物理信息神经网络（PINet）**的复振幅重建项目，用于无透镜成像 / 同轴全息相位恢复。

## 核心问题

给定一张衍射强度图像（无透镜成像系统采集），同时重建物平面的**振幅**和**相位**。这是一个病态逆问题，因为相位信息在强度测量中丢失了。

## 方法

项目使用**物理信息驱动的迭代深度学习**方法：

1. **复数 U-Net** 作为可学习的去噪器 / 正则化器（直接操作复数张量）
2. **角谱法 (Angular Spectrum Method, ASM)** 作为物理传播层嵌入网络
3. **交替迭代**：物理投影（利用已知传递函数更新估计）+ 学习去噪（U-Net 细化）
4. **损失函数** 同时约束振幅、相位的重建精度以及衍射图案的一致性

## 项目结构

```
PINet_cpx_cl/
├── src/                              # 核心源码
│   ├── config.py                     # 全局配置（物理参数、训练超参、路径、设备）
│   ├── model.py                      # PINet_cpx_v6 + 复数 U-Net + 实数 U-Net + CBAM 注意力
│   ├── complexFunctions.py           # 复数运算函数（矩阵乘、激活、池化、上采样、dropout 等）
│   ├── complexLayers.py              # 复数运算 nn.Module 封装（Conv2d、BatchNorm、ReLU 等）
│   ├── mydataset.py                  # 数据集类 + collate_fn（内含随机传播距离的 TF 计算）
│   ├── myloss.py                     # 损失函数（幅相损失 ComplexLoss_amp_phs、实虚部损失 ComplexLoss_re_im）
│   └── utilities.py                  # 物理传播（ASM）、衍射计算、PSNR / SSIM 指标、归一化
│
├── dataset_construction/             # 数据集构造流水线
│   ├── config.py                     # 流水线路径和超参
│   ├── crop_dataset.py               # ① 从参考复振幅网格裁剪 patch，生成 UNet 训练集
│   ├── model.py                      # 实数 U-Net（3 层 / 4 层，带 CBAM），训练振幅→相位映射
│   ├── train_unet.py                 # ② 训练 4 层 U-Net，定期保存 checkpoint + 对比图
│   ├── generate_dataset_unwrapped.py # ③ 自然图→灰度→直方图匹配→UNet 推理→PUMA 解缠绕→输出数据集
│   ├── utils.py                      # 数据 IO、直方图匹配、AmpPhsDataset
│   ├── puma_ho_py.py                 # PUMA-HO 相位解缠绕算法（图割 / 最小割，依赖 PyMaxflow）
│   └── README.md                     # 流水线详细说明
│
├── notebooks/                        # 训练与测试脚本
│   ├── train_new.ipynb               # PINet 训练（几何增强：随机翻转 + 90°旋转）
│   ├── test_single_real.ipynb        # 单模型在真实数据上的测试（支持多 epoch checkpoint 对比）
│   ├── test_single_sim.ipynb         # 单模型在仿真数据上的测试
│   ├── test_compare_real.ipynb       # 4 个模型在 4 个真实数据集上的对比测试
│   ├── test_compare_sim.ipynb        # 4 个模型在 12 个仿真数据文件上的对比测试
│   └── vix_train_data.ipynb          # 快速查看训练集振幅和相位
│
├── datasets/                         # 数据集目录（不纳入版本控制）
│   ├── pinet_1600/                   # 1600 张 RGB 自然图像 (256, 256)
│   ├── pinet_dataset_compared_*/     # 合成数据集（train/val/test, 各含 label_amp.npy + label_phs.npy）
│   └── data_simulation/              # 仿真复振幅 .mat 文件
│
├── real_data/                        # 真实实验数据（.mat，不纳入版本控制）
│   └── *_{type1,type2}_new.mat       # 内部键: Shift_Samples (衍射图), dist (距离), label (可选真值)
│
├── save_dir/                         # 训练好的模型和日志（不纳入版本控制）
│   └── model_saved_pinet_compared*/   # PINet 模型（compared1-4 对应不同训练配置）
│       ├── losses.txt                # 每 epoch 损失
│       ├── psnr.txt                  # 验证集 PSNR
│       └── training_log_*.txt        # 详细训练日志
│
└── requirements.txt                  # Python 依赖
```

## 数据流

### 训练流程

```
Ground Truth 复振幅 (amp + phs)
  │
  ├── ASM 前向传播 (随机 z ∈ [Z_LOW, Z_HIGH] mm) ──→ 衍射强度图（模型输入）
  │
  └── PINet 迭代重建:
        for i in fold_iters:
          1. ASM 前向: x → z_hat (传播到探测器面)
          2. 物理投影: z_hat → x_hat (用测量值替换振幅，保留相位)
          3. ASM 反向: x_hat → x (传播回物平面)
          4. U-Net 去噪: x → x_denoised
      输出: 预测复振幅 + 重建衍射图
  │
  └── 损失: MSE(pred_amp, gt_amp) + MSE(pred_phs, gt_phs) + MSE(y_rec, diff)
```

### 数据集构造流程

```
参考复振幅 (real .mat)
  │
  ├── 提取振幅 ──→ 作为直方图匹配的目标分布
  │
  └── 网格裁剪 → amp.npy + phs.npy ──→ 训练 UNet (振幅→相位)

自然图像 (1600 张)
  │
  ├── 转灰度
  ├── 直方图匹配到参考振幅分布
  ├── 随机混合 (匹配图 + 原图)
  ├── UNet 推理预测包裹相位
  ├── PUMA-HO 解缠绕得到连续相位
  └── 保存为 pinet_dataset/
      ├── train/ (1200 组) {label_amp.npy, label_phs.npy}
      ├── val/   (200 组)
      └── test/  (200 组)
```

## 快速开始

```bash
git clone git@github.com:Flowater1014/PINet_cpx_cl.git
cd PINet_cpx_cl
pip install -r requirements.txt
```

## 使用方式

### 训练

1. 准备好数据集放入 `datasets/`
2. 修改 `src/config.py`（可选）中的路径、超参、传播距离范围
3. 运行 `notebooks/train_new.ipynb`

训练过程中每 10 个 epoch 保存一次 checkpoint，并输出验证集 PSNR。

### 测试（仿真数据）

运行 `notebooks/test_single_sim.ipynb`，修改 `CHECKPOINT_PATH` 和 `Z_MM` 即可。

### 测试（真实数据）

运行 `notebooks/test_single_real.ipynb`，修改 `data_name` 切换数据集。支持：
- `distance_selection_mode='auto'`：自动在 [1.5, 3.0] mm 内选择最小距离
- `checkpoint_epochs`：指定要对比的 epoch 列表

### 多模型对比

运行 `notebooks/test_compare_real.ipynb` 或 `notebooks/test_compare_sim.ipynb`，结果保存到 `compare_results/`。

### 构造合成数据集

当缺乏真实复振幅标注数据时，使用 `dataset_construction/` 流水线从自然图像合成：

```bash
cd dataset_construction
python crop_dataset.py                 # ① 从参考复振幅裁剪 UNet 训练 patch
python train_unet.py                   # ② 训练振幅→相位 U-Net
python generate_dataset_unwrapped.py   # ③ 生成最终数据集
```

详见 `dataset_construction/README.md`。

## 配置系统

`src/config.py` 集中管理所有参数：

```python
from src.config import *

# 物理参数
WAVELENGTH = 532e-9       # 绿光 532 nm
PIXEL_SIZE = 1.67e-6      # 像素尺寸 (m)
IMAGE_SIZE = 256           # 图像尺寸

# 训练参数
BATCH_SIZE = 4
EPOCHS = 200
LEARNING_RATE = 1e-4
Z_LOW, Z_HIGH = 1.5, 3.0  # 传播距离范围 (mm)

# 模型参数
FOLD_ITERS = 5             # 物理迭代次数
```

Notebook 中 `from src.config import *` 后可按需覆盖任意变量。

## 关键依赖

- PyTorch >= 1.9.0（复数张量支持）
- NumPy, SciPy, scikit-image（PSNR/SSIM 指标）
- matplotlib（可视化）
- tqdm（进度条）
- PyMaxflow（PUMA 相位解缠绕，仅 dataset_construction 需要）
