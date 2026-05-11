"""
从参考复振幅中随机裁剪 patch，保存为 UNet 训练/验证/测试集。

输出目录结构:
  unet_dataset_JZX1/
  ├── meta.npy
  ├── train/  {amp.npy, phs.npy, coords.npy}
  ├── val/    {amp.npy, phs.npy, coords.npy}
  └── test/   {amp.npy, phs.npy, coords.npy}
"""

import os
import numpy as np
from scipy.io import loadmat
import config


def main():
    os.makedirs(config.UNET_DATA_DIR, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(config.UNET_DATA_DIR, split), exist_ok=True)

    # 1. 加载参考复振幅
    print(f"Loading {config.REF_MAT_PATH} -> key='{config.REF_KEY}'")
    data = loadmat(config.REF_MAT_PATH)
    O = np.squeeze(data[config.REF_KEY])
    H, W = O.shape
    print(f"Reference shape: {O.shape}, dtype: {O.dtype}")

    amp = np.abs(O).astype(np.float32)
    phs = np.angle(O).astype(np.float32)

    # 2. 生成随机裁剪坐标
    y_min = config.MARGIN
    y_max = H - config.MARGIN - config.PATCH_SIZE
    x_min = config.MARGIN
    x_max = W - config.MARGIN - config.PATCH_SIZE

    if y_min > y_max or x_min > x_max:
        raise ValueError(
            f"Image too small for patch_size={config.PATCH_SIZE} "
            f"with margin={config.MARGIN}: image ({H},{W})"
        )

    total = config.N_TRAIN + config.N_VAL + config.N_TEST
    rng = np.random.default_rng(config.CROP_SEED)

    coords = set()
    while len(coords) < total:
        y = int(rng.integers(y_min, y_max + 1))
        x = int(rng.integers(x_min, x_max + 1))
        coords.add((y, x))

    coords = np.array(list(coords), dtype=np.int32)
    rng.shuffle(coords)

    n_train = config.N_TRAIN
    n_val = config.N_VAL

    train_coords = coords[:n_train]
    val_coords = coords[n_train:n_train + n_val]
    test_coords = coords[n_train + n_val:]

    # 3. 裁剪并保存
    for split_name, split_coords in [
        ("train", train_coords),
        ("val", val_coords),
        ("test", test_coords),
    ]:
        out_dir = os.path.join(config.UNET_DATA_DIR, split_name)
        N = split_coords.shape[0]

        amp_p = np.empty((N, 1, config.PATCH_SIZE, config.PATCH_SIZE), dtype=np.float32)
        phs_p = np.empty((N, 1, config.PATCH_SIZE, config.PATCH_SIZE), dtype=np.float32)

        for i, (y, x) in enumerate(split_coords):
            amp_p[i, 0] = amp[y:y + config.PATCH_SIZE, x:x + config.PATCH_SIZE]
            phs_p[i, 0] = phs[y:y + config.PATCH_SIZE, x:x + config.PATCH_SIZE]

        np.save(os.path.join(out_dir, "amp.npy"), amp_p, allow_pickle=False)
        np.save(os.path.join(out_dir, "phs.npy"), phs_p, allow_pickle=False)
        np.save(os.path.join(out_dir, "coords.npy"), split_coords, allow_pickle=False)

        print(
            f"[{split_name}] {N} patches, "
            f"amp: [{amp_p.min():.4f}, {amp_p.max():.4f}], "
            f"phs: [{phs_p.min():.4f}, {phs_p.max():.4f}]"
        )

    # 4. 保存元信息
    meta = {
        "mat_path": config.REF_MAT_PATH,
        "key": config.REF_KEY,
        "original_shape": (int(H), int(W)),
        "patch_size": config.PATCH_SIZE,
        "margin": config.MARGIN,
        "counts": {"train": config.N_TRAIN, "val": config.N_VAL, "test": config.N_TEST},
        "seed": config.CROP_SEED,
    }
    np.save(os.path.join(config.UNET_DATA_DIR, "meta.npy"), meta, allow_pickle=True)

    print(f"\nDone. All saved to: {config.UNET_DATA_DIR}")


if __name__ == "__main__":
    main()
