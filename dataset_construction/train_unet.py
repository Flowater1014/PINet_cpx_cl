"""
训练四层 U-Net（UNet_4layer），仅训练集，无验证集。

用法:
  python train_unet_4layer.py

需要先运行 crop_dataset.py 生成训练数据。
checkpoint 保存到 checkpoints_JZX1/ 目录下，每 40 epoch 保存一次。
训练完成后自动生成不同 epoch 的预测对比图。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from model import UNet_4layer
from utils import AmpPhsDataset


def _clip_percentile(img, low=0.5, high=99.5):
    vmin, vmax = np.percentile(img, [low, high])
    return np.clip(img, vmin, vmax)


def compare_epochs(num_samples=4):
    """加载各 epoch checkpoint，对固定样本推理并生成对比图."""
    device = config.DEVICE
    ckpt_dir = config.CKPT_DIR_4LAYER

    epochs = [0, 40, 80, 120, 160, 200]
    ckpt_paths = [os.path.join(ckpt_dir, f"epoch_{e}.pt") for e in epochs]

    # 检查 checkpoint 是否存在
    missing = [p for p in ckpt_paths if not os.path.exists(p)]
    if missing:
        print(f"Missing checkpoints: {missing}")
        return

    # 加载样本（取验证集的前 num_samples 个，如果没验证集就用训练集）
    train_ds = AmpPhsDataset(config.UNET_DATA_DIR, "train")
    indices = np.linspace(0, len(train_ds) - 1, num_samples, dtype=int)
    samples_amp = []
    samples_phs = []
    for idx in indices:
        amp, phs = train_ds[idx]
        samples_amp.append(amp)
        samples_phs.append(phs)

    model = UNet_4layer(n_channels=1, n_classes=1, bilinear=False).to(device)

    # 对每个 checkpoint 推理
    all_preds = {}  # epoch -> list of preds
    for e, ckpt_path in zip(epochs, ckpt_paths):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        preds = []
        with torch.no_grad():
            for amp in samples_amp:
                x = amp.unsqueeze(0).to(device)  # (1, 1, H, W)
                p = model(x).cpu().numpy()[0, 0]
                preds.append(p)
        all_preds[e] = preds

    # 画图: num_samples 行, 2 + len(epochs) 列
    n_cols = 2 + len(epochs)  # Amp | Target | epoch_0 | ... | epoch_200
    fig, axes = plt.subplots(num_samples, n_cols, figsize=(3.5 * n_cols, 3.2 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    col_titles = ["Amp"] + ["Target Phase"] + [f"epoch_{e}" for e in epochs]

    for row in range(num_samples):
        amp = samples_amp[row][0].numpy()
        target = samples_phs[row][0].numpy()

        # Amp
        ax = axes[row, 0]
        ax.imshow(_clip_percentile(amp), cmap="gray")
        ax.set_title(col_titles[0] if row == 0 else "")
        ax.axis("off")

        # Target Phase
        ax = axes[row, 1]
        ax.imshow(_clip_percentile(target), cmap="gray")
        ax.set_title(col_titles[1] if row == 0 else "")
        ax.axis("off")

        # Predictions from each epoch
        for ci, e in enumerate(epochs):
            ax = axes[row, 2 + ci]
            pred = all_preds[e][row]
            ax.imshow(_clip_percentile(pred), cmap="gray")
            ax.set_title(col_titles[2 + ci] if row == 0 else "")
            ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(ckpt_dir, "compare_epochs.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Comparison figure saved: {save_path}")
    plt.close()


def main():
    os.makedirs(config.CKPT_DIR_4LAYER, exist_ok=True)
    device = config.DEVICE
    print(f"Device: {device}")

    # 1. 数据
    train_ds = AmpPhsDataset(config.UNET_DATA_DIR, "train", augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    print(f"Train: {len(train_ds)} samples")

    # 2. 模型
    model = UNet_4layer(n_channels=1, n_classes=1, bilinear=False).to(device)

    with torch.no_grad():
        tmp = torch.randn(2, 1, config.PATCH_SIZE, config.PATCH_SIZE, device=device)
        out = model(tmp)
        print(f"Sanity check: {tmp.shape} -> {out.shape}")

    # 3. 损失 & 优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    # 4. 保存初始状态 (epoch_0)
    init_path = os.path.join(config.CKPT_DIR_4LAYER, "epoch_0.pt")
    torch.save(model.state_dict(), init_path)
    print(f"Saved: {os.path.basename(init_path)} (initial state)")

    # 5. 训练
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n = 0

        pbar = tqdm(train_loader, leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            n += bs
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = total_loss / n
        print(f"Epoch {epoch:3d}/{config.EPOCHS}  train_mse={avg_loss:.6f}")

        # 每 40 个 epoch 保存一次
        if epoch % 40 == 0:
            ckpt_path = os.path.join(config.CKPT_DIR_4LAYER, f"epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  --> saved {os.path.basename(ckpt_path)}")

    # 6. 保存最后模型
    last_path = os.path.join(config.CKPT_DIR_4LAYER, "last.pt")
    torch.save(model.state_dict(), last_path)
    print(f"Saved: {last_path}")

    # 7. 生成 epoch 对比图
    print("\nGenerating epoch comparison figure ...")
    compare_epochs(num_samples=4)


if __name__ == "__main__":
    main()
