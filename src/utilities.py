import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os

from src.config import DEVICE, IMAGE_SIZE, TRANSFORM

# 兼容旧代码的导出名
device = DEVICE
transform = TRANSFORM
size = IMAGE_SIZE

# 前向传播
def ASM_forward(obj, TF):
    objFT = torch.fft.fftshift(torch.fft.fft2(obj))
    propfield = torch.fft.ifft2(torch.fft.ifftshift(objFT * TF))
    return propfield

# 计算衍射图像
def diff_compute(label_tensor, TF_torch):
    label_tensor = label_tensor.to(device)
    Det_field = ASM_forward(label_tensor, TF_torch)
    Real_meas = torch.abs(Det_field)
    return Real_meas

# 张量线性归一化
def normalize_tensor(tensor):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    return (tensor - tensor_min) / (tensor_max - tensor_min)

# 单张 PSNR
def compute_psnr_skimage_single(test_tensor, label_tensor):
    test_img = test_tensor.detach().cpu().numpy()
    label_img = label_tensor.detach().cpu().numpy()
    test_img = normalize_tensor(test_img)
    label_img = normalize_tensor(label_img)
    return psnr(test_img, label_img)

# 单张 SSIM
def compute_ssim_skimage_single(test_tensor, label_tensor):
    test_img = test_tensor.squeeze().detach().cpu().numpy()
    label_img = label_tensor.squeeze().detach().cpu().numpy()
    test_img = normalize_tensor(test_img)
    label_img = normalize_tensor(label_img)
    return ssim(test_img, label_img, win_size=7, data_range=1.0)
