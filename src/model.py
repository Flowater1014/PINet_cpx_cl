import torch
from torch import nn
import torch.nn.functional as F
from src.complexLayers import ComplexAvgPool2d, ComplexMaxPool2d, ComplexUpsample, ComplexUpsample2
from src.complexLayers import ComplexAdaptiveAvgPool2d, ComplexAdaptiveMaxPool2d
from src.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexReLU, ComplexLinear, ComplexSigmoid
from src.complexLayers import ComplexConvTranspose2d
from src.complexFunctions import complex_mul
from src.utilities import *
from src.CI_CDNet import CI_CDNet
# from complex_uformer import ComplexUformer
# from complex_uformer_without_timesteps import ComplexUformer_without_timesteps,ComplexUformer_without_timesteps_3layers


# v6版本的PINet_cpx，采用了通道注意力机制和空间注意力机制的UNet
class PINet_cpx_v6(nn.Module):
    def __init__(self, fold_iters=5, ratio=8):
        super(PINet_cpx_v6, self).__init__()
        
        self.fold_iters = fold_iters
        self.denoiser   = UNet_cpx_v4(1, 1, ratio)          # Complex UNet model

    def ASM_forward(self, obj, TF):
        objFT = torch.fft.fftshift(torch.fft.fft2(obj))
        propfield = torch.fft.ifft2(torch.fft.ifftshift(objFT*TF))

        return propfield

    def ASM_backward(self, propfield, TF):
        objFT = torch.fft.fftshift(torch.fft.fft2(propfield))
        obj = torch.fft.ifft2(torch.fft.ifftshift(objFT*torch.conj(TF)))

        return obj
        
    def forward(self, y, TF_hat):   
        x = torch.ones_like(y)

        for i in range(self.fold_iters):
            z_hat = self.ASM_forward(x, TF_hat)
            x_hat = self.ASM_backward(y * torch.exp(1j * torch.angle(z_hat)), TF_hat)
            x = self.denoiser(x_hat)
            # .detach()
        y_rec = torch.abs(self.ASM_forward(x, TF_hat))
            
        return x, y_rec
    
# v7版本的PINet_cpx，每次迭代采用不同的UNet，alpha = 1.0完全相信UNet，alpha = 0.0完全不用 U-Net，只做物理投影迭代。
class PINet_cpx_v7(nn.Module):
    def __init__(self, fold_iters=4, ratio=8, alpha=0.5):
        super(PINet_cpx_v7, self).__init__()

        self.fold_iters = fold_iters
        self.alpha = alpha

        self.denoisers = nn.ModuleList([
            UNet_cpx_v4(1, 1, ratio) for _ in range(fold_iters)
        ])

    def ASM_forward(self, obj, TF):
        objFT = torch.fft.fftshift(torch.fft.fft2(obj))
        propfield = torch.fft.ifft2(torch.fft.ifftshift(objFT * TF))
        return propfield

    def ASM_backward(self, propfield, TF):
        objFT = torch.fft.fftshift(torch.fft.fft2(propfield))
        obj = torch.fft.ifft2(torch.fft.ifftshift(objFT * torch.conj(TF)))
        return obj

    def forward(self, y, TF_hat):
        x = torch.ones_like(y, dtype=torch.complex64)

        for i in range(self.fold_iters):
            z_hat = self.ASM_forward(x, TF_hat)
            z_new = y * torch.exp(1j * torch.angle(z_hat))
            x_hat = self.ASM_backward(z_new, TF_hat)

            x_denoised = self.denoisers[i](x_hat)
            x = self.alpha * x_denoised + (1 - self.alpha) * x_hat

        y_rec = torch.abs(self.ASM_forward(x, TF_hat))

        return x, y_rec

# v6版本的PINet_cpx，每次迭代采用不同的UNet，alpha = 1.0完全相信UNet，alpha = 0.0完全不用 U-Net，只做物理投影迭代。
class PINet_cpx_CICDNet(nn.Module):
    def __init__(self, fold_iters=4):
        super(PINet_cpx_CICDNet, self).__init__()
        
        self.fold_iters = fold_iters
        self.denoiser   = CI_CDNet()          # Complex UNet model

    def ASM_forward(self, obj, TF):
        objFT = torch.fft.fftshift(torch.fft.fft2(obj))
        propfield = torch.fft.ifft2(torch.fft.ifftshift(objFT*TF))

        return propfield

    def ASM_backward(self, propfield, TF):
        objFT = torch.fft.fftshift(torch.fft.fft2(propfield))
        obj = torch.fft.ifft2(torch.fft.ifftshift(objFT*torch.conj(TF)))

        return obj
        
    def forward(self, y, TF_hat):   
        x = torch.ones_like(y)

        for i in range(self.fold_iters):
            z_hat = self.ASM_forward(x, TF_hat)
            x_hat = self.ASM_backward(y * torch.exp(1j * torch.angle(z_hat)), TF_hat)
            x = self.denoiser(x_hat)
            # .detach()
        y_rec = torch.abs(self.ASM_forward(x, TF_hat))
            
        return x, y_rec

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(256, 128, bilinear)
        self.up2 = Up(128, 64, bilinear)
        self.up3 = Up(64, 32, bilinear)
        # self.up4 = Up(64, 64, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x_input = x.clone()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        # x = self.up4(x, x1)
        logits = self.outc(x) + x_input
        return logits
    
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


## 新版UNet上下采样
class BlurPool(nn.Module):
    def __init__(self, channels, stride=2):
        super().__init__()
        # 5x5 low-pass kernel (same as in kornia/anti-alias networks)
        kernel = torch.tensor([
            [1, 4, 6, 4, 1],
            [4,16,24,16, 4],
            [6,24,36,24, 6],
            [4,16,24,16, 4],
            [1, 4, 6, 4, 1]
        ], dtype=torch.float32)
        kernel = kernel / kernel.sum()

        self.register_buffer('weight', kernel[None,None,:,:].repeat(channels,1,1,1))
        self.stride = stride
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.weight, stride=self.stride, padding=2, groups=self.channels)


class Down(nn.Module):
    """Anti-aliased downscaling: blur + conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.blurpool = BlurPool(in_channels, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.blurpool(x)    # anti-alias downsample
        x = self.double_conv(x)
        return x



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# 带有通道注意力和空间注意力机制的复数UNet，为v4
class UNet_cpx_v4(nn.Module):
    def __init__(self, n_channels, n_classes, ratio, bilinear=False):
        super(UNet_cpx_v4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear 

        self.inc = DoubleConv_cpx(n_channels, 32)
        self.down1 = Down_cpx(32, 64)
        self.down2 = Down_cpx(64, 128)
        self.down3 = Down_cpx(128, 256)
        self.cbam_block = cbam_block_cpx(in_channels=256, ratio=ratio)

        # 新版 Up_cpx： (up_channels, skip_channels, out_channels)
        self.up1 = Up_cpx(256, 128, bilinear)
        self.up2 = Up_cpx(128, 64, bilinear)
        self.up3 = Up_cpx(64, 32, bilinear)

        self.outc = OutConv_cpx(32, n_classes)


    def forward(self, x):
        x_input = x.clone()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.cbam_block(x4)  # 添加通道注意力机制
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x) + x_input
        return logits     
  
    

class DoubleConv_cpx(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv_cpx = nn.Sequential(
            ComplexConv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # ComplexBatchNorm2d(mid_channels),
            ComplexReLU(),
            ComplexConv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # ComplexBatchNorm2d(out_channels),
            ComplexReLU(),
        )

    def forward(self, x):
        return self.double_conv_cpx(x)

## 新版复数UNet上下采样
# -------- 复数 BlurPool：对实部/虚部分别做 depthwise blur + 下采样 --------
class BlurPool_cpx(nn.Module):
    def __init__(self, channels, stride=2):
        super().__init__()
        # 5x5 低通核，和实数版一样
        kernel = torch.tensor([
            [1, 4, 6, 4, 1],
            [4,16,24,16, 4],
            [6,24,36,24, 6],
            [4,16,24,16, 4],
            [1, 4, 6, 4, 1]
        ], dtype=torch.float32)
        kernel = kernel / kernel.sum()

        # depthwise 卷积，每个通道一个相同的 kernel
        self.register_buffer(
            "weight",
            kernel[None, None, :, :].repeat(channels, 1, 1, 1)
        )
        self.stride = stride
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]，complex
        xr = x.real
        xi = x.imag

        xr_blur = F.conv2d(
            xr, self.weight,
            stride=self.stride, padding=2,
            groups=self.channels
        )
        xi_blur = F.conv2d(
            xi, self.weight,
            stride=self.stride, padding=2,
            groups=self.channels
        )

        return torch.complex(xr_blur, xi_blur)


# -------- 新版 Down_cpx：BlurPool_cpx + DoubleConv_cpx --------
class Down_cpx(nn.Module):
    """Anti-aliased downscaling: complex blur + complex double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.blurpool = BlurPool_cpx(in_channels, stride=2)
        self.double_conv = DoubleConv_cpx(in_channels, out_channels)

    def forward(self, x):
        x = self.blurpool(x)
        x = self.double_conv(x)
        return x


class Up_cpx(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = ComplexUpsample2(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_cpx(in_channels, out_channels, in_channels // 2)
        else:
            self.up = ComplexConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_cpx(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: from deeper layer, x2: skip connection
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        # padding: [left, right, top, bottom]
        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv_cpx(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_cpx, self).__init__()
        self.conv = ComplexConv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


    
class ChannelAttention_cpx(nn.Module):
    def __init__(self, in_channels, ratio):
        super(ChannelAttention_cpx, self).__init__()
        self.avg_pool = ComplexAdaptiveAvgPool2d((1, 1))
        self.max_pool = ComplexAdaptiveMaxPool2d((1, 1))

        # 利用1x1卷积代替全连接
        #self.fc1   = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.fc1   = ComplexConv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = ComplexReLU()
        #self.fc2   = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.fc2   = ComplexConv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = ComplexSigmoid()

    def forward(self, x):
        # print(f'ChannelAttention_cpx x_input.shape:{x.shape}')
        x1 = self.avg_pool(x)
        x2 = self.fc1(x1)
        x3 = self.relu1(x2)
        avg_out = self.fc2(x3)      
        #avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        x1_1 = self.max_pool(x)     
        x2_2 = self.fc1(x1_1)
        x3_3 = self.relu1(x2_2)
        max_out = self.fc2(x3_3)  
        #max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        
        return self.sigmoid(out)
    

# new version
class SpatialAttention_cpx(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_cpx, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = ComplexConv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = ComplexSigmoid()

    def forward(self, x):
        x_real = torch.real(x)
        x_imag = torch.imag(x)

        avg_out_real = torch.mean(x_real, dim=1, keepdim=True)
        avg_out_imag = torch.mean(x_imag, dim=1, keepdim=True)

        max_out_real, _ = torch.max(x_real, dim=1, keepdim=True)
        max_out_imag, _ = torch.max(x_imag, dim=1, keepdim=True)
        
        avg_out_c = torch.complex(avg_out_real - avg_out_imag, avg_out_real + avg_out_imag)  # [B, 1, H, W]
        max_out_c = torch.complex(max_out_real - max_out_imag, max_out_real + max_out_imag)  # [B, 1, H, W]

        # 拼接两个通道组：[avg_real, max_real, avg_imag, max_imag]
        x_cat = torch.cat([avg_out_c, max_out_c], dim=1) # shape: [B, 2, H, W]
        x_cat = x_cat.to(x.dtype)  # 🔧 显式转换为 complex64，兼容后续操作
        
        x_attn = self.conv1(x_cat)  # conv1: nn.Conv2d(4, 1, ...)
        attn = self.sigmoid(x_attn)
        
        return x * attn
    
class cbam_block_cpx(nn.Module):
    def __init__(self, in_channels, ratio=8, kernel_size=7):
        super(cbam_block_cpx, self).__init__()
        self.channelattention = ChannelAttention_cpx(in_channels, ratio=ratio)
        self.spatialattention = SpatialAttention_cpx(kernel_size=kernel_size)
                          
    def forward(self, x):
        x = x * self.channelattention(x)
        x = self.spatialattention(x)
        return x


    

