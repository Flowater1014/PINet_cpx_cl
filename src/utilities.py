import torch
import torch as th
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
size = 256
Nx, Ny = (1072,1912) # JZX1 MXWKY1
Nx, Ny = (1080,1920) # MXWKY1
# Nx, Ny = (1048,1848) # USAF
Nx, Ny = (1048,1880)  # HSTA
# Nx, Ny = (size,size)
lambd = np.array([532e-9]) # Wavelength of light (in meters)
k = 2*np.pi/lambd # wave number
z = np.array([1.5e-3]) # Propagation distance (in meters)
z = np.array([2.744e-3]) # HSTA
# z = np.array([1.716e-3]) # USAF
dx = np.array([1.67e-6]) # pixelsize(in meters)
# dx = np.array([1.34e-6]) # New Real Data
Lx = Nx * dx
Ly = Ny * dx
x = np.linspace(-Lx/2,Lx/2-dx,Nx) # coordinate vector (in meters)
y = np.linspace(-Ly/2,Ly/2-dx,Ny) # coordinate vector (in meters)
X,Y = np.meshgrid(x,y)
fx = np.linspace(-1/(2*dx),1/(2*dx)-1/Lx,Nx) # Spatial frequency vector (in 1/meters)
fy = np.linspace(-1/(2*dx),1/(2*dx)-1/Ly,Ny) # Spatial frequency vector (in 1/meters)
FX,FY = np.meshgrid(fy,fx) # Coodinate grids (Fourier space)
TF = np.exp(1j*z*np.sqrt(k**2-(2*np.pi*FX)**2-(2*np.pi*FY)**2))
TF_torch = torch.from_numpy(TF).to(device).to(torch.complex64)

transform = transforms.Compose([
    transforms.Resize((size, size)),  # 调整大小到 256x256
    transforms.ToTensor()
    ])
to_pil = transforms.ToPILImage()

def diff(image, axis):
    '''
    Take the difference of different dimension(1~4) of images
    '''
    ndim = image.ndim
    if ndim == 3:    
        if axis == 0:
            return image[1:,:,:] - image[:-1,:,:]
        elif axis == 1:
            return image[:,1:,:] - image[:,:-1,:]
        elif axis == 2:
            return image[:,:,1:] - image[:,:,:-1]
        
    elif ndim == 2: 
        if axis == 0:
            return image[1:,:] - image[:-1,:]
        elif axis == 1:
            return image[:,1:] - image[:,:-1]
    elif ndim == 4:    
        if axis == 0:
            return image[1:,:,:,:] - image[:-1,:,:,:]
        elif axis == 1:
            return image[:,1:,:,:] - image[:,:-1,:,:]
        elif axis == 2:
            return image[:,:,1:,:] - image[:,:,:-1,:]
        elif axis == 3:
            return image[:,:,:,1:] - image[:,:,:,:-1]
    elif ndim == 1: 
        if axis == 0:
            return image[1:] - image[:-1]

                  
def _denoise_tv_chambolle_nd_torch(image, weight=0.1, eps=2.e-4, n_iter_max=200):
    """
    image : torch.tensor
        n-D input data to be denoised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
    n_iter_max : int, optional
        Maximal number of iterations used for the optimization.
    Returns
    -------
    out : torch.tensor
        Denoised array of floats.
    
    """    
    
    
    ndim = image.ndim
    pt = torch.zeros((image.ndim, ) + image.shape, dtype=image.dtype).to(image.device)
    gt = torch.zeros_like(pt)
    dt = torch.zeros_like(image)
    i = 0
    while i < n_iter_max:
       if i > 0:
           # dt will be the (negative) divergence of p
           dt = -pt.sum(0)
           slices_dt = [slice(None), ] * ndim
           slices_pt = [slice(None), ] * (ndim + 1)
           for ax in range(ndim):
               slices_dt[ax] = slice(1, None)
               slices_pt[ax+1] = slice(0, -1)
               slices_pt[0] = ax
               dt[tuple(slices_dt)] += pt[tuple(slices_pt)]
               slices_dt[ax] = slice(None)
               slices_pt[ax+1] = slice(None)
           out = image + dt
       else:
           out = image
       Et = torch.mul(dt,dt).sum()
       
       # gt stores the gradients of out along each axis
       # e.g. gt[0] is the first order finite difference along axis 0
       slices_gt = [slice(None), ] * (ndim + 1)
       for ax in range(ndim):
           slices_gt[ax+1] = slice(0, -1)
           slices_gt[0] = ax
           gt[tuple(slices_gt)] = diff(out, ax)
           slices_gt[ax+1] = slice(None)
            
       norm = torch.sqrt((gt ** 2).sum(axis=0)).unsqueeze(0)
       Et = Et + weight * norm.sum()
       tau = 1. / (2.*ndim)
       norm = norm * tau / weight
       norm = norm + 1.
       pt = pt - tau * gt
       pt = pt / norm
       Et = Et / float(image.view(-1).shape[0])
       if i == 0:
           E_init = Et
           E_previous = Et
       else:
           if torch.abs(E_previous - Et) < eps * E_init:
               break
           else:
               E_previous = Et
       i += 1
     
    return out


def denoise_tv_chambolle_torch(image, weight=0.1, eps=2.e-4, n_iter_max=200,
                         multichannel=False):
    
    """Perform total-variation denoising on n-dimensional images.
    Parameters
    ----------
    image : torch.tensor of ints, uints or floats
        Input data to be denoised. `image` can be of any numeric type,
        but it is cast into an torch.tensor of floats for the computation
        of the denoised image.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that
        determines the stop criterion. The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
    n_iter_max : int, optional
        Maximal number of iterations used for the optimization.
    multichannel : bool, optional
        Apply total-variation denoising separately for each channel. This
        option should be true for color images, otherwise the denoising is
        also applied in the channels dimension.
    Returns
    -------
    out : torch.tensor
        Denoised image.
    
    """
    # im_type = (image.numpy()).dtype
    # if not im_type.kind == 'f':
    #     image = image.type(torch.float64)
    #     image = image/torch.abs(image.max()+image.min())
        
    if multichannel:
        out = torch.zeros_like(image)
        for c in range(image.shape[-1]):
            out[...,c] = _denoise_tv_chambolle_nd_torch(image[..., c], weight, eps, n_iter_max)
    else:
        out = _denoise_tv_chambolle_nd_torch(image, weight, eps, n_iter_max)
    
    return out

def TV_denoising(y0, lamda, iteration=100):
    device = y0.device
    w, h, b  = y0.shape
    zh = torch.zeros([w, h-1, b], device=device, dtype=torch.float32)
    zv = torch.zeros([w-1, h, b], device=device, dtype=torch.float32)
    alpha = 5
    for it in range(iteration):
        x0h = y0 - dht_3d(zh)
        x0v = y0 - dvt_3d(zv)
        x0 = (x0h + x0v) / 2
        zh = clip(zh + 1/alpha*dh(x0), lamda/2)
        zv = clip(zv + 1/alpha*dv(x0), lamda/2)
    return x0

def TV_denoising3d(y0, lamda, iteration=100):
    device = y0.device
    # z = torch.zeros(y0.shape - [1, 1, 1], device=device, dtype=torch.float32)
    w, h, b  = y0.shape
    zh = torch.zeros([w, h-1, b], device=device, dtype=torch.float32)
    zv = torch.zeros([w-1, h, b], device=device, dtype=torch.float32)
    zt = torch.zeros([w, h, b-1], device=device, dtype=torch.float32)
    alpha = 5
    for it in range(iteration):
        x0h = y0 - dht_3d(zh)
        x0v = y0 - dvt_3d(zv)
        x0t = y0 - dtt_3d(zt)
        x0 = (x0h + x0v + x0t) / 3
        zh = clip(zh + 1/alpha*dh(x0), lamda/2)
        zv = clip(zv + 1/alpha*dv(x0), lamda/2)
        zt = clip(zt + 1/alpha*dt(x0), lamda/2)
    return x0

def clip(x, thres):
    return torch.clamp(x, min=-thres, max=thres)

def dht_3d(x):
    return torch.cat([-x[:,0:1,:], x[:,:-1,:]-x[:,1:,:], x[:,-1:,:]], 1)

def dvt_3d(x):
    return torch.cat([-x[0:1,:,:], x[:-1,:,:]-x[1:,:,:], x[-1:,:,:]], 0)

def dtt_3d(x):
    return torch.cat([-x[:,:,0:1], x[:,:,:-1]-x[:,:,1:], x[:,:,-1:]], 2)

def dh(x):
    return x[:,1:,:]-x[:,:-1,:]

def dv(x):
    return x[1:,:,:] - x[:-1,:,:]

def dt(x):
    return x[:,:,1:] - x[:,:,:-1]


# 前向传播
def ASM_forward(obj, TF):
    objFT = torch.fft.fftshift(torch.torch.fft.fft2(obj))
    propfield = torch.fft.ifft2(torch.fft.ifftshift(objFT*TF))

    return propfield

# 计算衍射图像
def diff_compute(label_tensor, TF_torch):
    # 前向传播
    label_tensor = label_tensor.to(device)
    Det_field = ASM_forward(label_tensor, TF_torch)
    Real_meas = torch.abs(Det_field)
    return Real_meas


# 把张量线性归一化
def normalize_tensor(tensor):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return normalized_tensor

def tensor_numpy_show(img_diff, img_cpl):
    # 将张量转换为 NumPy 数组并归一化到 [0, 1] 的范围
    img_diff_np = img_diff.squeeze().detach().cpu().numpy()  # 去掉通道维度并转为 NumPy 数组
    img_cpl_amp = torch.abs(img_cpl).squeeze().detach().cpu().numpy()  # 实部
    img_cpl_phs = torch.angle(img_cpl).squeeze().detach().cpu().numpy()  # 虚部
    
    # 线性归一化到 [0, 1]
    img_diff_np = normalize_tensor(img_diff_np)
    img_cpl_amp = normalize_tensor(img_cpl_amp)
    img_cpl_phs = normalize_tensor(img_cpl_phs)

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    ax[0].imshow(img_diff_np, cmap='gray') 
    ax[0].set_title('Diffraction Input')
    ax[0].axis('off')  
    ax[1].imshow(img_cpl_amp, cmap='gray')
    ax[1].set_title('Amp of OBJ')
    ax[1].axis('off') 
    ax[2].imshow(img_cpl_phs, cmap='gray') 
    ax[2].set_title('Phs of OBJ')
    ax[2].axis('off')  
    plt.show()


# 计算PSNR
def compute_psnr(test_tensor, label_tensor):
    test_tensor = test_tensor.squeeze()
    label_tensor = label_tensor.squeeze()
    test_tensor = normalize_tensor(test_tensor)
    label_tensor = normalize_tensor(label_tensor)
    assert test_tensor.shape == label_tensor.shape, "Input tensors must have the same shape."
    max_val = max(test_tensor.max().item(), label_tensor.max().item())
    mse = F.mse_loss(test_tensor, label_tensor)
    if mse.item() == 0:
        return float('inf')  # 返回无穷大
    psnr = 10 * torch.log10((max_val ** 2) / mse)
    return psnr.item()

# 分别算16张算平均值
def compute_psnr_skimage(test_tensor, label_tensor):
    num = test_tensor.size(0)
    psnr_values = []
    # 对每一张图像计算 PSNR
    for i in range(num):
        test_img = test_tensor[i].detach().cpu().numpy()
        label_img = label_tensor[i].detach().cpu().numpy()
        # 计算单张图像的 PSNR
        psnr_value = psnr(test_img, label_img)
        psnr_values.append(psnr_value)
    # 计算平均 PSNR
    avg_psnr = np.mean(psnr_values)

    return avg_psnr


# 计算单张PSNR
def compute_psnr_skimage_single(test_tensor, label_tensor):

    test_img = test_tensor.detach().cpu().numpy()
    label_img = label_tensor.detach().cpu().numpy()
    test_img = normalize_tensor(test_img)
    label_img = normalize_tensor(label_img)    
    psnr_value = psnr(test_img, label_img)

    return psnr_value

# 计算单张ssim
def compute_ssim_skimage_single(test_tensor, label_tensor):

    test_img = test_tensor.squeeze().detach().cpu().numpy()
    label_img = label_tensor.squeeze().detach().cpu().numpy()
    test_img = normalize_tensor(test_img)
    label_img = normalize_tensor(label_img)   
    ssim_value = ssim(test_img, label_img,win_size=7, data_range=1.0)


    return ssim_value

# 计算整个测试集的平均 PSNR
def compute_average_psnr(dataloader_test, model, device, TF_torch):
    total_psnr_amp = 0
    total_psnr_phs = 0
    total_samples = 0

    with torch.no_grad():  # 禁用梯度计算以节省内存
        for diff, obj in dataloader_test:
            diff = diff.to(device)
            obj = obj.to(device)

            # 使用模型得到预测值
            pred, y_rec = model(diff, TF_torch)

            # 提取振幅和相位
            pred_amp = torch.abs(pred)
            pred_phs = torch.angle(pred)
            obj_amp = torch.abs(obj)
            obj_phs = torch.angle(obj)

            # 归一化
            pred_amp = normalize_tensor(pred_amp)
            pred_phs = normalize_tensor(pred_phs)

            # 计算 PSNR（使用 skimage 方法）
            psnr_amp = compute_psnr_skimage(pred_amp, obj_amp)
            psnr_phs = compute_psnr_skimage(pred_phs, obj_phs)

            # 累计 PSNR 和样本数量
            total_psnr_amp += psnr_amp
            total_psnr_phs += psnr_phs
            total_samples += 1

    # 计算平均 PSNR
    avg_psnr_amp = total_psnr_amp / total_samples
    avg_psnr_phs = total_psnr_phs / total_samples

    return avg_psnr_amp, avg_psnr_phs

# 计算整个测试集的平均 PSNR
def compute_average_psnr_2(dataloader_test, model_1, model_2, device, TF_torch):
    total_psnr_amp = 0
    total_psnr_phs = 0
    total_samples = 0

    with torch.no_grad():  # 禁用梯度计算以节省内存
        for diff, obj in dataloader_test:
            diff = diff.to(device)
            obj = obj.to(device)

            # 使用模型得到预测值
            pred, y_rec = model_1(diff, TF_torch)
            pred, y_rec = model_2(diff, pred, TF_torch)

            # 提取振幅和相位
            pred_amp = torch.abs(pred)
            pred_phs = torch.angle(pred)
            obj_amp = torch.abs(obj)
            obj_phs = torch.angle(obj)

            # 归一化
            pred_amp = normalize_tensor(pred_amp)
            pred_phs = normalize_tensor(pred_phs)

            # 计算 PSNR（使用 skimage 方法）
            psnr_amp = compute_psnr_skimage(pred_amp, obj_amp)
            psnr_phs = compute_psnr_skimage(pred_phs, obj_phs)

            # 累计 PSNR 和样本数量
            total_psnr_amp += psnr_amp
            total_psnr_phs += psnr_phs
            total_samples += 1

    # 计算平均 PSNR
    avg_psnr_amp = total_psnr_amp / total_samples
    avg_psnr_phs = total_psnr_phs / total_samples

    return avg_psnr_amp, avg_psnr_phs




# 两个PIL转复数tensor
def PIL_to_cpl_tensor(amp_img,phs_img):
    amp_tensor = transform(amp_img)
    phs_tensor = transform(phs_img)
    cpl_tensor = amp_tensor * torch.exp(1j * phs_tensor)  # 复数标签
    return cpl_tensor

# 复数tensor转两个PIL
def cpl_tensor_to_PIL(cpl_tensor):
    amp_tensor = torch.abs(cpl_tensor)
    phs_tensor = torch.angle(cpl_tensor)
    amp_pil = to_pil(amp_tensor)
    phs_pil = to_pil(phs_tensor)
    return amp_pil,phs_pil  

# 显示图片(衍射图像、目标振幅、目标相位)
def tensor_pil_show(img_diff,img_cpl):
    to_pil = transforms.ToPILImage()
    img_diff_pil = to_pil(img_diff)
    img_cpl_amp = torch.abs(img_cpl)
    img_cpl_phs = torch.angle(img_cpl)
    img_cpl_amp_pil = to_pil(img_cpl_amp)
    img_cpl_phs_pil = to_pil(img_cpl_phs)

    fig,ax = plt.subplots(1,3,figsize=(12,6))
    ax[0].imshow(img_diff_pil)
    ax[0].set_title('Diffraction Input')
    ax[1].imshow(img_cpl_amp_pil)
    ax[1].set_title('Amp of OBJ')
    ax[2].imshow(img_cpl_phs_pil)
    ax[2].set_title('Phs of OBJ')   

    plt.show() 


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]                        # B x half
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


