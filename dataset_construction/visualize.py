"""
可视化工具函数，可直接导入 notebook 中调用。

用法示例:
  from visualize import check_crop_samples, check_histogram_match, check_prediction
  check_crop_samples(4)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import config
from model import UNet
from utils import (
    AmpPhsDataset,
    load_ref_complex,
    load_gray_as_float,
    histogram_match,
)


def check_crop_samples(n=4, split="train"):
    """显示从参考数据中裁剪的振幅-相位配对样本."""
    ds = AmpPhsDataset(config.UNET_DATA_DIR, split)
    idxs = np.random.choice(len(ds), size=n, replace=False)

    fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for row, idx in enumerate(idxs):
        amp, phs = ds[idx]

        im0 = axes[row, 0].imshow(amp[0], cmap="gray")
        axes[row, 0].set_title(f"Amplitude (idx={idx})")
        axes[row, 0].axis("off")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im1 = axes[row, 1].imshow(phs[0], cmap="gray")
        axes[row, 1].set_title(f"Phase (idx={idx})")
        axes[row, 1].axis("off")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def check_histogram_match(n=4):
    """对比自然图像直方图匹配前后的效果."""
    ref_complex = load_ref_complex(config.REF_MAT_PATH, config.REF_KEY)
    ref_amp = np.abs(ref_complex)

    idxs = np.random.choice(config.NATURAL_IMG_NUM, size=n, replace=False) + 1

    fig, axes = plt.subplots(n, 3, figsize=(14, 3.5 * n))

    for row, idx in enumerate(idxs):
        gray = load_gray_as_float(idx, config.NATURAL_IMG_DIR)
        matched = histogram_match(gray, ref_amp)

        axes[row, 0].imshow(gray, cmap="gray")
        axes[row, 0].set_title(f"Original gray ({idx})")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(matched, cmap="gray")
        axes[row, 1].set_title("Histogram matched")
        axes[row, 1].axis("off")

        axes[row, 2].hist(gray.ravel(), bins=100, alpha=0.5, label="original", density=True)
        axes[row, 2].hist(matched.ravel(), bins=100, alpha=0.5, label="matched", density=True)
        axes[row, 2].hist(ref_amp.ravel(), bins=100, alpha=0.5, label="ref amp", density=True)
        axes[row, 2].legend(fontsize=8)
        axes[row, 2].set_title("Distribution")

    plt.tight_layout()
    plt.show()


def check_prediction(n=4, ckpt_name="best.pt"):
    """加载模型 → 推理 → 展示振幅、预测相位、分布."""
    device = config.DEVICE

    ckpt_path = os.path.join(config.CKPT_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    model = UNet(1, 1, bilinear=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    ref_complex = load_ref_complex(config.REF_MAT_PATH, config.REF_KEY)
    ref_amp = np.abs(ref_complex)

    idxs = np.random.choice(config.NATURAL_IMG_NUM, size=n, replace=False) + 1

    fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n))

    for row, idx in enumerate(idxs):
        gray = load_gray_as_float(idx, config.NATURAL_IMG_DIR)
        amp_matched = histogram_match(gray, ref_amp)

        x = torch.from_numpy(amp_matched[None, None]).to(device)
        with torch.no_grad():
            pred_phase = model(x).cpu().numpy()[0, 0]

        axes[row, 0].imshow(amp_matched, cmap="gray")
        axes[row, 0].set_title(f"Constructed Amp ({idx})")
        axes[row, 0].axis("off")

        im = axes[row, 1].imshow(pred_phase, cmap="gray")
        axes[row, 1].set_title("Predicted Phase")
        axes[row, 1].axis("off")
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)

        axes[row, 2].hist(pred_phase.ravel(), bins=100)
        axes[row, 2].set_title("Phase distribution")

    plt.tight_layout()
    plt.show()
