"""
完整数据集生成流水线：
  自然图像 → 转灰度 → 直方图匹配（以参考振幅分布为目标）→ U-Net 预测相位 → 保存

输出:
  datasets/synth_dataset_v1/
  ├── train/  {label_amp.npy, label_phs.npy}   (1200, 256, 256)
  ├── val/    {label_amp.npy, label_phs.npy}   (200, 256, 256)
  └── test/   {label_amp.npy, label_phs.npy}   (200, 256, 256)

需要先运行 train_unet.py 得到模型权重。
"""

import os
import numpy as np
import torch
from tqdm import tqdm

import config
from model import UNet
from utils import load_ref_complex, load_gray_as_float, histogram_match


def save_split(split_name, amp_data, phs_data):
    split_dir = os.path.join(config.OUTPUT_DATASET_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    amp_path = os.path.join(split_dir, "label_amp.npy")
    phs_path = os.path.join(split_dir, "label_phs.npy")
    np.save(amp_path, amp_data.astype(np.float32), allow_pickle=False)
    np.save(phs_path, phs_data.astype(np.float32), allow_pickle=False)
    print(f"  [{split_name}] {amp_data.shape[0]} samples, "
          f"amp [{amp_data.min():.4f}, {amp_data.max():.4f}], "
          f"phs [{phs_data.min():.4f}, {phs_data.max():.4f}]")


def main():
    os.makedirs(config.OUTPUT_DATASET_DIR, exist_ok=True)
    device = config.DEVICE
    print(f"Device: {device}")

    # 1. 加载模型
    ckpt_path = os.path.join(config.CKPT_DIR, "best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(config.CKPT_DIR, "last.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"No checkpoint found in {config.CKPT_DIR}. "
                f"Run train_unet.py first."
            )
    print(f"Loading checkpoint: {ckpt_path}")

    model = UNet(1, 1, bilinear=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print("Model loaded.")

    # 2. 加载参考振幅分布（作为直方图匹配的目标）
    print(f"Loading reference: {config.REF_MAT_PATH}")
    ref_complex = load_ref_complex(config.REF_MAT_PATH, config.REF_KEY)
    ref_amp = np.abs(ref_complex)
    print(f"Reference amplitude: shape={ref_amp.shape}, "
          f"range=[{ref_amp.min():.4f}, {ref_amp.max():.4f}]")

    # 3. 预分配数组
    N = config.NATURAL_IMG_NUM
    H, W = 256, 256
    amp_all = np.empty((N, H, W), dtype=np.float32)
    phs_all = np.empty((N, H, W), dtype=np.float32)

    # 4. 逐张处理
    print(f"Processing {N} images ...")
    for i in tqdm(range(1, N + 1)):
        gray = load_gray_as_float(i, config.NATURAL_IMG_DIR)
        amp_matched = histogram_match(gray, ref_amp)

        x = torch.from_numpy(amp_matched[None, None]).to(device)  # (1,1,H,W)
        with torch.no_grad():
            phase = model(x).cpu().numpy()[0, 0]  # (H,W)

        amp_all[i - 1] = amp_matched
        phs_all[i - 1] = phase

    # 5. 切分并保存
    n_train = config.N_TRAIN_GEN
    n_val = config.N_VAL_GEN
    n_test = config.N_TEST_GEN
    assert n_train + n_val + n_test == N, \
        f"N_TRAIN_GEN + N_VAL_GEN + N_TEST_GEN ({n_train+n_val+n_test}) != 总数 ({N})"

    save_split("train", amp_all[:n_train], phs_all[:n_train])
    save_split("val", amp_all[n_train:n_train + n_val], phs_all[n_train:n_train + n_val])
    save_split("test", amp_all[n_train + n_val:], phs_all[n_train + n_val:])

    print(f"\nSaved to: {config.OUTPUT_DATASET_DIR}")


if __name__ == "__main__":
    main()
