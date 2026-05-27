# dataset_construction

使用 U-Net 和 PUMA 相位解缠绕，从自然图像构造振幅-相位配对数据集的流水线。

## 整体流程

```
JZX1 label (real_data/JZX1_type1_new.mat, 复振幅 1074×1919)
  │
  ├── PUMA 解缠绕 + 网格裁剪 ──→ unet_dataset_JZX1/train/ (500 patches)
  │                               └── AmpPhsDataset → train UNet_4layer
  │                                                    ↓
  │                                         checkpoints_JZX1/
  │                                         ├── epoch_0.pt ... epoch_200.pt
  │                                         ├── last.pt
  │                                         └── compare_epochs.png
  │
  └── 1600 张自然图 ──→ 灰度 → 直方图匹配 → UNet 推理 → PUMA 解缠绕
                          ↓
                     pinet_dataset_compared_6/
                     ├── train/ (1200 组)
                     ├── val/   (200 组)
                     └── test/  (200 组)
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `crop_dataset.py` | PUMA 解缠 + 网格裁剪 JZX1 相位，生成 UNet 训练集 |
| `train_unet_4layer.py` | 训练 4 层 U-Net，每 40 epoch 保存一次，训练完自动生成对比图 |
| `generate_dataset_unwrapped.py` | 用训练好的 UNet 推理 1600 张自然图 + PUMA 解缠，产出最终数据集 |
| `compare_unwrap.py` | 展示 JZX1 相位 PUMA 解缠前后对比 |
| `model.py` | UNet / UNet_4layer / CBAM / Down / Up 等网络组件 |
| `utils.py` | 数据加载、直方图匹配、AmpPhsDataset |
| `config.py` | 路径和超参数配置 |
| `puma_ho_py.py` | PUMA-HO 相位解缠绕算法的 Python 实现（依赖 PyMaxflow） |
| `visualize.py` | 可视化辅助函数 |

## 按步骤运行

### 1. 生成 UNet 训练集

```bash
python crop_dataset.py
```

- 对 JZX1 label 全图做 PUMA 解缠
- 网格裁剪 25×20=500 个 256×256 patch（margin=80）
- 输出 `unet_dataset_JZX1/train/amp.npy` + `phs.npy`

### 2. 训练 UNet

```bash
python train_unet_4layer.py
```

- 在 500 个 patch 上训练 UNet_4layer（200 epoch，MSE 损失）
- Checkpoint 保存到 `checkpoints_JZX1/`（epoch_0, 40, 80, 120, 160, 200, last.pt）
- 训练完自动生成 `checkpoints_JZX1/compare_epochs.png`

### 3. 生成最终数据集

```bash
python generate_dataset_unwrapped.py
```

- 加载 `checkpoints_JZX1/last.pt`
- 逐张处理 1600 张自然图：灰度 → 直方图匹配 → 随机混合 → UNet 推理 → PUMA 解缠
- 输出 `datasets/pinet_dataset_compared_6/{train,val,test}/`

### 4. （可选）查看 PUMA 解缠效果

```bash
python compare_unwrap.py
```

- 展示 JZX1 label 相位 PUMA 解缠前后对比图

## 数据集说明

### pinet_dataset_compared_6 (v6)

**文件结构:**

```
datasets/pinet_dataset_compared_6/
├── train/   (1200 组)
│   ├── label_amp.npy  — 直方图匹配 + 随机混合后的振幅 (N, 256, 256)
│   └── label_phs.npy  — PUMA 解缠绕后的连续相位 (N, 256, 256)
├── val/     (200 组)
│   ├── label_amp.npy
│   └── label_phs.npy
└── test/    (200 组)
    ├── label_amp.npy
    └── label_phs.npy
```

**生成流程:**

```
自然图 RGB → 灰度
  → 直方图匹配到 JZX1 振幅分布
  → 随机混合 (λ × 匹配图 + (1-λ) × 原图, λ ~ U[0,1])
  → UNet_4layer (checkpoints_JZX1/last.pt) 预测包裹相位
  → PUMA-HO 解缠绕 → 保存
```

**关键参数:** 1600 张图, 切分 1200/200/200, UNet_4layer, PUMA-HO (p=1, quantized=yes)

## 依赖

- PyTorch, numpy, scipy, matplotlib, tqdm
- Pillow (图像读写)
- PyMaxflow (PUMA 解缠绕依赖)
