"""
加载预裁剪的振幅-相位 npy 数据，训练 U-Net。

用法:
  python train_unet.py

需要先运行 crop_dataset.py 生成训练数据。
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from model import UNet
from utils import AmpPhsDataset


def run_one_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n = 0

    pbar = tqdm(loader, leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        if train:
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / max(1, n)


def main():
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    device = config.DEVICE
    print(f"Device: {device}")

    # 1. 数据
    train_ds = AmpPhsDataset(config.UNET_DATA_DIR, "train")
    val_ds = AmpPhsDataset(config.UNET_DATA_DIR, "val")
    test_ds = AmpPhsDataset(config.UNET_DATA_DIR, "test")

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    print(f"Train: {len(train_ds)} samples")
    print(f"Val:   {len(val_ds)} samples")
    print(f"Test:  {len(test_ds)} samples")

    # 2. 模型
    model = UNet(n_channels=1, n_classes=1, bilinear=False).to(device)

    # 前向检查
    with torch.no_grad():
        tmp = torch.randn(2, 1, config.PATCH_SIZE, config.PATCH_SIZE, device=device)
        out = model(tmp)
        print(f"Sanity check: {tmp.shape} -> {out.shape}")

    # 3. 损失函数 & 优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    # 4. 训练
    best_val_loss = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = run_one_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss = run_one_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        print(f"Epoch {epoch:3d}/{config.EPOCHS}  "
              f"train_mse={train_loss:.6f}  val_mse={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config.CKPT_DIR, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  --> saved best.pt (val_mse={val_loss:.6f})")

    # 5. 保存最后模型
    last_path = os.path.join(config.CKPT_DIR, "last.pt")
    torch.save(model.state_dict(), last_path)
    print(f"Saved: {last_path}")

    # 6. 测试集评估
    test_loss = run_one_epoch(
        model, test_loader, criterion, optimizer, device, train=False
    )
    print(f"Test  MSE: {test_loss:.6f}")


if __name__ == "__main__":
    main()
