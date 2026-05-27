"""
从参考复振幅中按网格裁剪 patch，保存为 UNet 训练集。

流程:
  1. 加载 Eout_beewings_unwrapped.mat -> P_HSSA_unwrap 复振幅
  2. 提取振幅 + 解缠绕相位（数据已解缠绕，无需 PUMA）
  3. 均匀网格裁剪 (MARGIN_TOP/BOTTOM/LEFT/RIGHT, N_V×N_H patches)
  4. 保存为 train/amp.npy, train/phs.npy

输出目录结构:
  unet_dataset_Beewings/
  └── train/
      ├── amp.npy   (500, 1, 256, 256)
      └── phs.npy   (500, 1, 256, 256)
"""

import os
import numpy as np
from scipy.io import loadmat

import config


def crop_complex_patches_grid(
    img,
    patch_size=256,
    margin_top=300,
    margin_bottom=300,
    margin_left=200,
    margin_right=200,
    num_y=25,
    num_x=20,
):
    """在有效区域内均匀网格裁剪复数 patch。

    参数
    ----------
    img : np.ndarray, shape (H, W)
        二维复数图像。
    patch_size : int
        patch 边长。
    margin_{top,bottom,left,right} : int
        各方向排除的边缘像素数。
    num_y : int
        y 方向网格数量。
    num_x : int
        x 方向网格数量。

    返回
    -------
    patches : np.ndarray, shape (num_y*num_x, patch_size, patch_size)
    coords : list of tuple
        每个 patch 的左上角坐标 (y, x)。
    """
    H, W = img.shape

    y_min = margin_top
    y_max = H - margin_bottom - patch_size
    x_min = margin_left
    x_max = W - margin_right - patch_size

    if y_min >= y_max or x_min >= x_max:
        raise ValueError(
            f"有效区域为空！H={H}, W={W}, patch_size={patch_size}, "
            f"margins=({margin_top},{margin_bottom},{margin_left},{margin_right})"
        )

    y_positions = np.linspace(y_min, y_max, num_y, dtype=np.int32)
    x_positions = np.linspace(x_min, x_max, num_x, dtype=np.int32)

    print(f"Image: H={H}, W={W}  |  patch_size={patch_size}")
    print(f"  y range: [{y_min}, {y_max}], step ~{(y_max - y_min) / max(num_y - 1, 1):.1f}")
    print(f"  x range: [{x_min}, {x_max}], step ~{(x_max - x_min) / max(num_x - 1, 1):.1f}")
    print(f"  Grid: {num_y} (y) × {num_x} (x) = {num_y * num_x} patches")

    total = num_y * num_x
    patches = np.empty((total, patch_size, patch_size), dtype=img.dtype)
    coords = []

    idx = 0
    for vy in y_positions:
        for vx in x_positions:
            patches[idx] = img[vy:vy + patch_size, vx:vx + patch_size]
            coords.append((int(vy), int(vx)))
            idx += 1

    assert idx == total
    return patches, coords


def main():
    os.makedirs(config.UNET_DATA_DIR, exist_ok=True)
    train_dir = os.path.join(config.UNET_DATA_DIR, "train")
    os.makedirs(train_dir, exist_ok=True)

    # 1. 加载参考复振幅
    print(f"Loading {config.REF_MAT_PATH} -> key='{config.REF_KEY}'")
    data = loadmat(config.REF_MAT_PATH)
    O = np.squeeze(data[config.REF_KEY])
    H, W = O.shape
    print(f"Reference shape: {O.shape}, dtype: {O.dtype}")

    # 2. 根据实际 shape 自适应网格参数
    if H == 4001 and W == 5423:
        num_y, num_x = 20, 25   # H=4001 用 20, W=5423 用 25
    elif H == 5423 and W == 4001:
        num_y, num_x = 25, 20   # H=5423 用 25, W=4001 用 20
    else:
        num_y, num_x = config.N_V, config.N_H
        print(f"  Unexpected shape, using num_y={num_y}, num_x={num_x}")

    # 3. 提取振幅和相位（数据已解缠绕，无需 PUMA）
    amp = np.abs(O).astype(np.float32)
    phs = np.angle(O).astype(np.float32)

    print(f"Amplitude : [{amp.min():.4f}, {amp.max():.4f}]")
    print(f"Phase     : [{phs.min():.4f}, {phs.max():.4f}] rad")

    # 4. 均匀网格裁剪（直接对复数数组操作）
    complex_patches, coords = crop_complex_patches_grid(
        O,
        patch_size=config.PATCH_SIZE,
        margin_top=config.MARGIN_TOP,
        margin_bottom=config.MARGIN_BOTTOM,
        margin_left=config.MARGIN_LEFT,
        margin_right=config.MARGIN_RIGHT,
        num_y=num_y,
        num_x=num_x,
    )

    # 5. 转为振幅和相位 patch
    total = len(complex_patches)
    amp_patches = np.empty((total, 1, config.PATCH_SIZE, config.PATCH_SIZE), dtype=np.float32)
    phs_patches = np.empty((total, 1, config.PATCH_SIZE, config.PATCH_SIZE), dtype=np.float32)

    for i in range(total):
        amp_patches[i, 0] = np.abs(complex_patches[i])
        phs_patches[i, 0] = np.angle(complex_patches[i])

    # 6. 保存
    np.save(os.path.join(train_dir, "amp.npy"), amp_patches, allow_pickle=False)
    np.save(os.path.join(train_dir, "phs.npy"), phs_patches, allow_pickle=False)

    print(f"\n[train] {total} patches saved.")
    print(f"  amp: [{amp_patches.min():.4f}, {amp_patches.max():.4f}]")
    print(f"  phs: [{phs_patches.min():.2f}, {phs_patches.max():.2f}] rad")
    print(f"Done. Saved to: {train_dir}")


if __name__ == "__main__":
    main()
