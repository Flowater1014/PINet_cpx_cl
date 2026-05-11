import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def load_ref_complex(mat_path, key):
    """从 .mat 文件中加载复振幅数组."""
    from scipy.io import loadmat
    data = loadmat(mat_path)
    return np.squeeze(data[key])


def extract_amp_phase(z):
    """从复数数组中提取振幅和相位."""
    return np.abs(z), np.angle(z)


class AmpPhsDataset(Dataset):
    """加载预裁剪的振幅-相位配对 npy 文件."""

    def __init__(self, root, split="train"):
        self.amp = np.load(os.path.join(root, split, "amp.npy"))  # (N,1,H,W)
        self.phs = np.load(os.path.join(root, split, "phs.npy"))  # (N,1,H,W)

    def __len__(self):
        return self.amp.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.amp[idx]).float()
        y = torch.from_numpy(self.phs[idx]).float()
        return x, y


def load_gray_as_float(idx, img_dir):
    """读取 RGB 图像并转为灰度图，返回 [0, 1] 范围的 float32 数组."""
    path = os.path.join(img_dir, f"{idx}_256.jpg")
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def histogram_match(source, ref_values):
    """
    将 source 的直方图匹配到 ref_values 的分布。

    参数
    ----------
    source : np.ndarray, shape (H, W)
        待匹配的图像。
    ref_values : np.ndarray, shape (H', W') 或 (N,)
        目标分布。

    返回
    -------
    matched : np.ndarray, shape (H, W)
        匹配后的图像。
    """
    src_flat = source.ravel()
    ref_flat = ref_values.ravel() if ref_values.ndim > 1 else ref_values

    src_sorted = np.sort(src_flat)
    ref_sorted = np.sort(ref_flat)

    # 将 source 每个像素的分位数映射到 ref 的相同分位数
    mapped = np.interp(
        np.linspace(0, 1, len(src_sorted)),
        np.linspace(0, 1, len(ref_sorted)),
        ref_sorted,
    )

    src_order = np.argsort(src_flat)
    result = np.empty_like(src_flat, dtype=np.float32)
    result[src_order] = mapped

    return result.reshape(source.shape).astype(np.float32)
