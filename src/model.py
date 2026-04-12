import torch
from torch import nn
import torch.nn.functional as F
from complexLayers import ComplexAvgPool2d, ComplexMaxPool2d, ComplexUpsample, ComplexUpsample2
from complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexReLU, ComplexLinear, ComplexSigmoid
from complexLayers import ComplexConvTranspose2d
from complexFunctions import complex_mul
from utilities import *
# from complex_uformer import ComplexUformer
# from complex_uformer_without_timesteps import ComplexUformer_without_timesteps,ComplexUformer_without_timesteps_3layers


# v6版本的PINet_cpx，采用了通道注意力机制和空间注意力机制的UNet
class PINet_cpx_v6(nn.Module):
    def __init__(self, fold_iters=5, ratio=8):
        super(PINet_cpx_v6, self).__init__()
        
        self.fold_iters = fold_iters
        self.denoiser   = UNet_cpx_v4(1, 1, ratio)          # Complex UNet model

    def ASM_forward(self, obj, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(obj))
        propfield = torch.fft.ifft2(torch.fft.ifftshift(objFT*TF))

        return propfield

    def ASM_backward(self, propfield, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(propfield))
        obj = torch.fft.ifft2(torch.fft.ifftshift(objFT*torch.conj(TF)))

        return obj
        
    def forward(self, y, TF_hat):   
        x = torch.ones_like(y)

        for i in range(self.fold_iters):
            z_hat = self.ASM_forward(x, TF_hat)
            x_hat = self.ASM_backward(y * torch.exp(1j * torch.angle(z_hat)), TF_hat).detach()
            x = self.denoiser(x_hat).detach()
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


## 旧版UNet上下采样
# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)

# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

    
# 复数UNet，输入cpx tensor，输出cpx tensor
class UNet_cpx_v1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_cpx_v1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv_cpx(n_channels, 32)
        self.down1 = Down_cpx(32, 64)
        self.down2 = Down_cpx(64, 128)
        self.down3 = Down_cpx(128, 256)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        self.up1 = Up_cpx(256, 128, bilinear)
        self.up2 = Up_cpx(128, 64, bilinear)
        self.up3 = Up_cpx(64, 32, bilinear)
        # self.up4 = Up(64, 64, bilinear)
        self.outc = OutConv_cpx(32, n_classes)

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

# 带有通道注意力机制的复数UNet，为v2
class UNet_cpx_v2(nn.Module):
    def __init__(self, n_channels, n_classes, ratio, bilinear=False):
        super(UNet_cpx_v2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv_cpx(n_channels, 32)
        self.down1 = Down_cpx(32, 64)
        self.down2 = Down_cpx(64, 128)
        self.down3 = Down_cpx(128, 256)
        self.ChannelAttention = ChannelAttention_cpx(in_channels=256, ratio=ratio)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        self.up1 = Up_cpx(256, 128, bilinear)
        self.up2 = Up_cpx(128, 64, bilinear)
        self.up3 = Up_cpx(64, 32, bilinear)
        # self.up4 = Up(64, 64, bilinear)
        self.outc = OutConv_cpx(32, n_classes)

    def forward(self, x):
        x_input = x.clone()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.ChannelAttention(x4)  # 添加通道注意力机制
        # x5 = self.down4(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        # x = self.up4(x, x1)
        logits = self.outc(x) + x_input
        return logits    

# 带有空间注意力机制的复数UNet，为v3
class UNet_cpx_v3(nn.Module):
    def __init__(self, n_channels, n_classes, ratio, bilinear=False):
        super(UNet_cpx_v3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv_cpx(n_channels, 32)
        self.down1 = Down_cpx(32, 64)
        self.down2 = Down_cpx(64, 128)
        self.down3 = Down_cpx(128, 256)
        self.SpatialAttention = SpatialAttention_cpx(kernel_size=7)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        self.up1 = Up_cpx(256, 128, bilinear)
        self.up2 = Up_cpx(128, 64, bilinear)
        self.up3 = Up_cpx(64, 32, bilinear)
        # self.up4 = Up(64, 64, bilinear)
        self.outc = OutConv_cpx(32, n_classes)

    def forward(self, x):
        x_input = x.clone()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = x4 * self.SpatialAttention(x4)  # 添加通道注意力机制
        # x5 = self.down4(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        # x = self.up4(x, x1)
        logits = self.outc(x) + x_input
        return logits   

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
    
# 带有通道注意力和空间注意力机制的复数UNet，为v4
class UNet_cpx_v4_pre(nn.Module):
    def __init__(self, n_channels, n_classes, ratio, bilinear=False):
        super(UNet_cpx_v4_pre, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear 

        self.inc = DoubleConv_cpx(n_channels, 32)
        self.down1 = Down_cpx_pre(32, 64)
        self.down2 = Down_cpx_pre(64, 128)
        self.down3 = Down_cpx_pre(128, 256)
        self.cbam_block = cbam_block_cpx(in_channels=256, ratio=ratio)

        # 新版 Up_cpx： (up_channels, skip_channels, out_channels)
        self.up1 = Up_cpx_pre(256, 128, bilinear)
        self.up2 = Up_cpx_pre(128, 64, bilinear)
        self.up3 = Up_cpx_pre(64, 32, bilinear)

        self.outc = OutConv_cpx(32, n_classes)


    def forward(self, x):
        x_input = x.clone()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.cbam_block(x4)  # 添加通道注意力机制
        # x5 = self.down4(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        # x = self.up4(x, x1)
        logits = self.outc(x) + x_input
        return logits   


# 带有通道注意力机制的复数UNet，为v5，相比v2改了窗口参数
class UNet_cpx_v5(nn.Module):
    def __init__(self, n_channels, n_classes, ratio, bilinear=False):
        super(UNet_cpx_v5, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv_cpx(n_channels, 32)
        self.down1 = Down_cpx(32, 64)
        self.down2 = Down_cpx(64, 128)
        self.down3 = Down_cpx(128, 256)
        self.ChannelAttention = ChannelAttention_cpx(in_channels=256, ratio=ratio)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        self.up1 = Up_cpx(256, 128, bilinear)
        self.up2 = Up_cpx(128, 64, bilinear)
        self.up3 = Up_cpx(64, 32, bilinear)
        # self.up4 = Up(64, 64, bilinear)
        self.outc = OutConv_cpx(32, n_classes)

    def forward(self, x):
        x_input = x.clone()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4_n = self.ChannelAttention(x4)
        x4 = x4 * x4_n   # 添加通道注意力机制
        # x5 = self.down4(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        # x = self.up4(x, x1)
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
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

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


# 旧版复数UNet上下采样
class Down_cpx_pre(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            ComplexMaxPool2d(2),
            DoubleConv_cpx(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up_cpx_pre(nn.Module):
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
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv_cpx(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_cpx, self).__init__()
        self.conv = ComplexConv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# 输入采集振幅，输出目标幅度    
class PINet_amp(nn.Module):
    def __init__(self, fold_iters=5):
        super(PINet_amp, self).__init__()
        
        self.fold_iters = fold_iters
        self.denoiser   = UNet(1, 1)       # UNet model
        
    def ASM_forward(self, obj, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(obj))
        propfield = torch.fft.ifft2(torch.fft.ifftshift(objFT*TF))

        return propfield

    def ASM_backward(self, propfield, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(propfield))
        obj = torch.fft.ifft2(torch.fft.ifftshift(objFT*torch.conj(TF)))

        return obj
        
    def forward(self, y, TF_hat):   
        x = torch.ones_like(y)
        for i in range(self.fold_iters):
            z_hat = self.ASM_forward(x, TF_hat)
            x_hat = torch.abs(self.ASM_backward(y * torch.exp(1j * torch.angle(z_hat)), TF_hat))
            x = self.denoiser(x_hat.float())
            
        y_rec = torch.abs(self.ASM_forward(x, TF_hat))
            
        return x, y_rec

class PINet_phs(nn.Module):
    def __init__(self, fold_iters=5):
        super(PINet_phs, self).__init__()
        
        self.fold_iters = fold_iters
        self.denoiser   = UNet(1, 1)       # UNet model
        
    def ASM_forward(self, obj, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(obj))
        propfield = torch.fft.ifft2(torch.fft.ifftshift(objFT*TF))

        return propfield

    def ASM_backward(self, propfield, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(propfield))
        obj = torch.fft.ifft2(torch.fft.ifftshift(objFT*torch.conj(TF)))

        return obj
        
    def forward(self, y, TF_hat):   
        x_phase = torch.ones_like(y)
        x = torch.exp(1j * x_phase)
        for i in range(self.fold_iters):
            z_hat = self.ASM_forward(x, TF_hat)
            x_hat = torch.angle(self.ASM_backward(y * torch.exp(1j * torch.angle(z_hat)), TF_hat))
            x_phase = self.denoiser(x_hat.float())
            x = torch.exp(1j * x_phase)
            
        y_rec = torch.abs(self.ASM_forward(x, TF_hat))
            
        return x_phase, y_rec


class PINet_cpx_v1(nn.Module):
    def __init__(self, fold_iters=5):
        super(PINet_cpx_v1, self).__init__()
        
        self.fold_iters = fold_iters
        self.denoiser_amp   = UNet(1, 1)       # UNet model
        self.denoiser_phs   = UNet(1, 1)       # UNet model
        
    def ASM_forward(self, obj, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(obj))
        propfield = torch.fft.ifft2(torch.fft.ifftshift(objFT*TF))

        return propfield

    def ASM_backward(self, propfield, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(propfield))
        obj = torch.fft.ifft2(torch.fft.ifftshift(objFT*torch.conj(TF)))

        return obj
        
    def forward(self, y, TF_hat):   
        x = torch.ones_like(y)
        for i in range(self.fold_iters):
            z_hat = self.ASM_forward(x, TF_hat)
            x_hat = self.ASM_backward(y * torch.exp(1j * torch.angle(z_hat)), TF_hat)
            x_amp = self.denoiser_amp(torch.abs(x_hat).float())
            x_phase = self.denoiser_phs(torch.angle(x_hat).float())          
            x = x_amp * torch.exp(1j * x_phase)
            
        y_rec = torch.abs(self.ASM_forward(x, TF_hat))
            
        return x, y_rec


class PINet_cpx_v2(nn.Module):
    def __init__(self, fold_iters=5):
        super(PINet_cpx_v2, self).__init__()
        
        self.fold_iters = fold_iters
        self.denoiser_re   = UNet(1, 1)       # UNet model
        self.denoiser_im   = UNet(1, 1)       # UNet model
        
    def ASM_forward(self, obj, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(obj))
        propfield = torch.fft.ifft2(torch.fft.ifftshift(objFT*TF))

        return propfield

    def ASM_backward(self, propfield, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(propfield))
        obj = torch.fft.ifft2(torch.fft.ifftshift(objFT*torch.conj(TF)))

        return obj
        
    def forward(self, y, TF_hat):   
        x = torch.ones_like(y)
        for i in range(self.fold_iters):
            z_hat = self.ASM_forward(x, TF_hat)
            x_hat = self.ASM_backward(y * torch.exp(1j * torch.angle(z_hat)), TF_hat)
            x_re = self.denoiser_re(torch.real(x_hat).float())
            x_im = self.denoiser_im(torch.imag(x_hat).float())          
            x = torch.complex(x_re, x_im)
            
        y_rec = torch.abs(self.ASM_forward(x, TF_hat))
            
        return x, y_rec
    

class PINet_cpx_v3(nn.Module):
    def __init__(self, fold_iters=5):
        super(PINet_cpx_v3, self).__init__()
        
        self.fold_iters = fold_iters
        self.denoiser   = UNet_cpx_v1(1, 1)          # Complex UNet model

    def ASM_forward(self, obj, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(obj))
        propfield = torch.fft.ifft2(torch.fft.ifftshift(objFT*TF))

        return propfield

    def ASM_backward(self, propfield, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(propfield))
        obj = torch.fft.ifft2(torch.fft.ifftshift(objFT*torch.conj(TF)))

        return obj
        
    def forward(self, y, TF_hat):   
        x = torch.ones_like(y)

        for i in range(self.fold_iters):
            z_hat = self.ASM_forward(x, TF_hat)
            x_hat = self.ASM_backward(y * torch.exp(1j * torch.angle(z_hat)), TF_hat)
            x = self.denoiser(x_hat)
            
        y_rec = torch.abs(self.ASM_forward(x, TF_hat))
            
        return x, y_rec


# v4版本的PINet_cpx，采用了通道注意力机制的UNet
class PINet_cpx_v4(nn.Module):
    def __init__(self, fold_iters=5, ratio=8):
        super(PINet_cpx_v4, self).__init__()
        
        self.fold_iters = fold_iters
        self.denoiser   = UNet_cpx_v2(1, 1, ratio)          # Complex UNet model

    def ASM_forward(self, obj, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(obj))
        propfield = torch.fft.ifft2(torch.fft.ifftshift(objFT*TF))

        return propfield

    def ASM_backward(self, propfield, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(propfield))
        obj = torch.fft.ifft2(torch.fft.ifftshift(objFT*torch.conj(TF)))

        return obj
        
    def forward(self, y, TF_hat):   
        x = torch.ones_like(y)

        for i in range(self.fold_iters):
            z_hat = self.ASM_forward(x, TF_hat)
            x_hat = self.ASM_backward(y * torch.exp(1j * torch.angle(z_hat)), TF_hat)
            x = self.denoiser(x_hat)
            
        y_rec = torch.abs(self.ASM_forward(x, TF_hat))
            
        return x, y_rec

    
# v5版本的PINet_cpx，采用了空间注意力机制的UNet
class PINet_cpx_v5(nn.Module):
    def __init__(self, fold_iters=5, ratio=8):
        super(PINet_cpx_v5, self).__init__()
        
        self.fold_iters = fold_iters
        self.denoiser   = UNet_cpx_v3(1, 1, ratio)          # Complex UNet model

    def ASM_forward(self, obj, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(obj))
        propfield = torch.fft.ifft2(torch.fft.ifftshift(objFT*TF))

        return propfield

    def ASM_backward(self, propfield, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(propfield))
        obj = torch.fft.ifft2(torch.fft.ifftshift(objFT*torch.conj(TF)))

        return obj
        
    def forward(self, y, TF_hat):   
        x = torch.ones_like(y)

        for i in range(self.fold_iters):
            z_hat = self.ASM_forward(x, TF_hat)
            x_hat = self.ASM_backward(y * torch.exp(1j * torch.angle(z_hat)), TF_hat)
            x = self.denoiser(x_hat)
            
        y_rec = torch.abs(self.ASM_forward(x, TF_hat))
            
        return x, y_rec
    

    

# v6版本的PINet_cpx，采用了通道注意力机制和空间注意力机制的UNet
class PINet_cpx_v6_pre(nn.Module):
    def __init__(self, fold_iters=5, ratio=8):
        super(PINet_cpx_v6_pre, self).__init__()
        
        self.fold_iters = fold_iters
        self.denoiser   = UNet_cpx_v4_pre(1, 1, ratio)          # Complex UNet model

    def ASM_forward(self, obj, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(obj))
        propfield = torch.fft.ifft2(torch.fft.ifftshift(objFT*TF))

        return propfield

    def ASM_backward(self, propfield, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(propfield))
        obj = torch.fft.ifft2(torch.fft.ifftshift(objFT*torch.conj(TF)))

        return obj
        
    def forward(self, y, TF_hat):   
        x = torch.ones_like(y)

        for i in range(self.fold_iters):
            z_hat = self.ASM_forward(x, TF_hat)
            x_hat = self.ASM_backward(y * torch.exp(1j * torch.angle(z_hat)), TF_hat)
            x = self.denoiser(x_hat)
            
        y_rec = torch.abs(self.ASM_forward(x, TF_hat))
            
        return x, y_rec
    
# v6版本的PINet_cpx，添加了相位解缠绕
class PINet_cpx_v6_unwrap(nn.Module):
    def __init__(self, fold_iters=5, ratio=8):
        super(PINet_cpx_v6_unwrap, self).__init__()
        
        self.fold_iters = fold_iters
        self.denoiser   = UNet_cpx_v4(1, 1, ratio)          # Complex UNet model

    def ASM_forward(self, obj, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(obj))
        propfield = torch.fft.ifft2(torch.fft.ifftshift(objFT*TF))

        return propfield

    def ASM_backward(self, propfield, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(propfield))
        obj = torch.fft.ifft2(torch.fft.ifftshift(objFT*torch.conj(TF)))

        return obj
    
    # ---------- utils: wrap/grad/div ----------
    @staticmethod
    def _wrap_to_pi(phi):
        # 映射到 (-pi, pi]
        two_pi = 2 * torch.pi
        return torch.remainder(phi + torch.pi, two_pi) - torch.pi

    @staticmethod
    def _grad_wrap(phi):
        """
        前向差分并 wrap 到 (-pi,pi]
        phi: [B, C, H, W]
        返回:
          dx: [B, C, H, W-1]  (沿 W 方向差分)
          dy: [B, C, H-1, W]  (沿 H 方向差分)
        """
        wrap = PINet_cpx_v6_unwrap._wrap_to_pi
        dx = wrap(phi[:, :, :, 1:] - phi[:, :, :, :-1])  # W 向
        dy = wrap(phi[:, :, 1:, :] - phi[:, :, :-1, :])  # H 向
        return dx, dy


    @staticmethod
    def _div_backward(dx, dy, H, W):
        """
        把 dx, dy 的散度（后向差分）组回 [B, C, H, W]
        dx: [B, C, H, W-1], dy: [B, C, H-1, W]
        """
        B, C = dx.shape[0], dx.shape[1]
        device, dtype = dx.device, dx.dtype
        div = torch.zeros((B, C, H, W), dtype=dtype, device=device)

        # x 方向（W）
        div[:, :, :, 0]    += dx[:, :, :, 0]
        if W > 2:
            div[:, :, :, 1:-1] += (dx[:, :, :, 1:] - dx[:, :, :, :-1])
        div[:, :, :, -1]   -= dx[:, :, :, -1]

        # y 方向（H）
        div[:, :, 0, :]    += dy[:, :, 0, :]
        if H > 2:
            div[:, :, 1:-1, :] += (dy[:, :, 1:, :] - dy[:, :, :-1, :])
        div[:, :, -1, :]   -= dy[:, :, -1, :]

        return div


    @staticmethod
    def unwrap2d_ls(phi):
        """
        最小二乘二维相位解缠绕（Poisson + FFT，周期边界）
        输入:  phi [B,C,H,W]
        输出:  phi_unw [B,C,H,W]，零均值
        """
        assert phi.dim() == 4, "phi should be [B,C,H,W]"
        B, C, H, W = phi.shape
        dtype, device = phi.dtype, phi.device

        # 1) 包裹梯度
        dx, dy = PINet_cpx_v6_unwrap._grad_wrap(phi)  # [B,C,H,W-1], [B,C,H-1,W]

        # 2) 散度
        div = PINet_cpx_v6_unwrap._div_backward(dx, dy, H, W)  # [B,C,H,W]

        # 3) 频域 Poisson（周期边界）
        kx = torch.arange(0, W, device=device, dtype=dtype).view(1,1,1,W)
        ky = torch.arange(0, H, device=device, dtype=dtype).view(1,1,H,1)
        lam_x = 2 - 2*torch.cos(2*torch.pi * kx / W)     # [1,1,1,W]
        lam_y = 2 - 2*torch.cos(2*torch.pi * ky / H)     # [1,1,H,1]
        lam = lam_x + lam_y                               # [1,1,H,W]

        Div = torch.fft.fft2(div, dim=(-2,-1))
        lam[..., 0, 0] = 1.0
        Phi = Div / lam
        phi_unw = torch.fft.ifft2(Phi, dim=(-2,-1)).real

        # 4) 去常数
        phi_unw = phi_unw - phi_unw.mean(dim=(-2,-1), keepdim=True)
        return phi_unw

        
    def forward(self, y, TF_hat):   
        x = torch.ones_like(y)

        for i in range(self.fold_iters):
            z_hat = self.ASM_forward(x, TF_hat)
            x_hat = self.ASM_backward(y * torch.exp(1j * torch.angle(z_hat)), TF_hat)
            x = self.denoiser(x_hat)

        # ===== 最小二乘二维相位解缠绕（LS） =====
        amp = torch.abs(x)
        phi = torch.angle(x)                          # [-pi, pi]
        phi_unw = PINet_cpx_v6_unwrap.unwrap2d_ls(phi)  # 连续相位（零均值）
        phi_unw = normalize_tensor(phi_unw)

        # （可选）按振幅掩膜进行常数锚定，避免弱信号区域影响中值
        # mask = (amp > 0.05 * amp.amax(dim=(-2,-1), keepdim=True))
        # med  = torch.median(phi_unw[mask]) if mask.any() else phi_unw.mean()
        # phi_unw = phi_unw - 2*torch.pi*torch.round(med / (2*torch.pi))

        x = torch.polar(amp, phi_unw)                 # 用解缠相位重组复数
        
        y_rec = torch.abs(self.ASM_forward(x, TF_hat))
        return x, y_rec


    
# v8版本的PINet_cpx，采用了通道注意力机制的UNet
class PINet_cpx_v8(nn.Module):
    def __init__(self, fold_iters=5, ratio=8):
        super(PINet_cpx_v8, self).__init__()
        
        self.fold_iters = fold_iters
        self.denoiser   = UNet_cpx_v5(1, 1, ratio)          # Complex UNet model

    def ASM_forward(self, obj, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(obj))
        propfield = torch.fft.ifft2(torch.fft.ifftshift(objFT*TF))

        return propfield

    def ASM_backward(self, propfield, TF):
        objFT = torch.fft.fftshift(torch.torch.fft.fft2(propfield))
        obj = torch.fft.ifft2(torch.fft.ifftshift(objFT*torch.conj(TF)))

        return obj
        
    def forward(self, y, TF_hat):   
        x = torch.ones_like(y)

        for i in range(self.fold_iters):
            z_hat = self.ASM_forward(x, TF_hat)
            x_hat = self.ASM_backward(y * torch.exp(1j * torch.angle(z_hat)), TF_hat)
            x = self.denoiser(x_hat)
            
        y_rec = torch.abs(self.ASM_forward(x, TF_hat))
            
        return x, y_rec
    
    
    
    
    
    
# --- 复数 <-> 两通道 工具 ---
def c2two(x_c: torch.Tensor) -> torch.Tensor:
    """complex [B,1,H,W] -> float [B,2,H,W]"""
    x_c = x_c.squeeze(1)
    return torch.stack([x_c.real, x_c.imag], dim=1).to(torch.float32)

def two2c(x_2: torch.Tensor) -> torch.Tensor:
    """float [B,2,H,W] -> complex [B,1,H,W]"""
    return torch.complex(x_2[:, 0], x_2[:, 1]).unsqueeze(1) 

# --- 你已有的时间步嵌入（保持一致）---
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# --- 假定 ComplexUformer 已经按我们之前建议：in_chans=2, out_channel=2, forward(x, timesteps, lq=None, mask=None) 返回 x+y ---
# from your_module import ComplexUformer

# class PINet_cpx_uformer(nn.Module):
#     """
#     使用 ComplexUformer 作为复数去噪器/修正器。
#     y: 传入的是“衍射振幅” (sensor amplitude)，形状 [B,1,H,W] 或 [B,H,W]
#     TF_hat: 角谱/ASM 的传递函数，复数 [B,H,W] 或 [H,W]（会广播）
#     size: 输入图像边长 H=W=size
#     fold_iters: 折叠次数（迭代步数，建议与 timestep 一起使用）
#     """
#     def __init__(
#         self,
#         size = (128,128),
#         fold_iters: int = 8,
#         u_embed_dim: int = 12,
#         win_size: int = 4
#     ):
#         super().__init__()
#         self.size = size
#         self.fold_iters = fold_iters

#         # ---- 实例化 ComplexUformer（确保 out_channel=2）----
#         self.denoiser = ComplexUformer(
#             img_size=size, in_chans=1, dd_in=1,
#             embed_dim=u_embed_dim, win_size=win_size,
#             depths=[1,1,1,1,1,1,1,1,1],
#             num_heads=[1,1,1,1,1,1,1,1,1],
#             drop_path_rate=0.05,
#             cond_lq=False   # 使用条件 lq（低质量先验）
#         )

#         # 把 ComplexUformer 里用到的时间 MLP 接口绑进来（或直接让其内部调用）
#         self.emb_dim = self.denoiser.embed_dim
#         self.time_embed = self.denoiser.time_embed  # 直接复用其 MLP

#     # ----- ASM 前/后向 -----
#     @staticmethod
#     def ASM_forward(obj, TF):
#         objFT = torch.fft.fftshift(torch.fft.fft2(obj))
#         return torch.fft.ifft2(torch.fft.ifftshift(objFT * TF))

#     @staticmethod
#     def ASM_backward(propfield, TF):
#         objFT = torch.fft.fftshift(torch.fft.fft2(propfield))
#         return torch.fft.ifft2(torch.fft.ifftshift(objFT * torch.conj(TF)))

#     def forward(self, y, TF_hat):

#         B = y.shape[0]
#         H,W = self.size
#         device = y.device
#         assert H % 16 == 0 and W % 16 == 0, "H,W 需要被 16 整除（4 次下采样）。"
#         x = torch.ones(B, 1, H, W, dtype=torch.complex64, device=device)

#         # ---- 物面低质量先验 g0----
#         y_c = y * torch.exp(1j * torch.zeros_like(y))
#         # print(f'y_c.shape:{y_c.shape}')
#         g0_2 = c2two(y_c)
#         # print(f'g0_2.shape:{g0_2.shape}')

#         # ---- 迭代折叠 ----
#         for i in range(self.fold_iters):
#             # 物 -> 像
#             z_hat = self.ASM_forward(x, TF_hat)  # [B,1,H,W] complex
#             z_new = y * torch.exp(1j * torch.angle(z_hat))  # [B,H,W] complex
#             # print(f'z_new = y * torch.exp(1j * torch.angle(z_hat)) z_new.shape:{z_new.shape}')
#             x_hat = self.ASM_backward(z_new, TF_hat)  # [B,H,W] complex
#             # print(f'x_hat = self.ASM_backward(z_new, TF_hat) x_hat.shape:{x_hat.shape}')
#             x_hat_2 = c2two(x_hat)  # [B,2,H,W]
#             # print(f'x_hat_2 = c2two(x_hat) x_hat_2.shape:{x_hat_2.shape}')

#             # 时间步（0..fold_iters-1）
#             t = torch.full((B,), 1, dtype=torch.long, device=device)
#             x_den_2 = self.denoiser(x_hat_2, timesteps=t)  # [B,2,H,W]
#             # print(f'x_den_2.shape:{x_den_2.shape}')
#             x = two2c(x_den_2)  # 回到 complex 物面
#             # print(f'x = two2c(x_den_2) x.shape:{x.shape}')

#         # ---- 输出：最终物面复场 + 像面重建振幅 ----
#         y_rec = torch.abs(self.ASM_forward(x, TF_hat))  # [B,H,W] float

#         return x, y_rec

    
# class PINet_cpx_uformer_without_timesteps(nn.Module):
#     """
#     使用 ComplexUformer 作为复数去噪器/修正器。
#     y: 传入的是“衍射振幅” (sensor amplitude)，形状 [B,1,H,W] 或 [B,H,W]
#     TF_hat: 角谱/ASM 的传递函数，复数 [B,H,W] 或 [H,W]（会广播）
#     size: 输入图像边长 H=W=size
#     fold_iters: 折叠次数（迭代步数，建议与 timestep 一起使用）
#     """
#     def __init__(
#         self,
#         size = (128,128),
#         fold_iters: int = 8,
#         u_embed_dim: int = 12,
#         win_size: int = 4
#     ):
#         super().__init__()
#         self.size = size
#         self.fold_iters = fold_iters

#         # ---- 实例化 ComplexUformer（确保 out_channel=2）----
#         self.denoiser = ComplexUformer_without_timesteps(
#             img_size=size, in_chans=1, dd_in=1,
#             embed_dim=u_embed_dim, win_size=win_size,
#             depths=[1,1,1,1,1,1,1,1,1],
#             num_heads=[1,1,1,1,1,1,1,1,1],
#             drop_path_rate=0.05,
#             cond_lq=False   # 使用条件 lq（低质量先验）
#         )

#         # 把 ComplexUformer 里用到的时间 MLP 接口绑进来（或直接让其内部调用）
#         self.emb_dim = self.denoiser.embed_dim
#         self.time_embed = self.denoiser.time_embed  # 直接复用其 MLP

#     # ----- ASM 前/后向 -----
#     @staticmethod
#     def ASM_forward(obj, TF):
#         objFT = torch.fft.fftshift(torch.fft.fft2(obj))
#         return torch.fft.ifft2(torch.fft.ifftshift(objFT * TF))

#     @staticmethod
#     def ASM_backward(propfield, TF):
#         objFT = torch.fft.fftshift(torch.fft.fft2(propfield))
#         return torch.fft.ifft2(torch.fft.ifftshift(objFT * torch.conj(TF)))

#     def forward(self, y, TF_hat):

#         B = y.shape[0]
#         H,W = self.size
#         device = y.device
#         assert H % 16 == 0 and W % 16 == 0, "H,W 需要被 16 整除（4 次下采样）。"
#         x = torch.ones(B, 1, H, W, dtype=torch.complex64, device=device)

#         # ---- 物面低质量先验 g0----
#         y_c = y * torch.exp(1j * torch.zeros_like(y))
#         # print(f'y_c.shape:{y_c.shape}')
#         g0_2 = c2two(y_c)
#         # print(f'g0_2.shape:{g0_2.shape}')

#         # ---- 迭代折叠 ----
#         for i in range(self.fold_iters):
#             # 物 -> 像
#             z_hat = self.ASM_forward(x, TF_hat)  # [B,1,H,W] complex
#             z_new = y * torch.exp(1j * torch.angle(z_hat))  # [B,H,W] complex
#             # print(f'z_new = y * torch.exp(1j * torch.angle(z_hat)) z_new.shape:{z_new.shape}')
#             x_hat = self.ASM_backward(z_new, TF_hat)  # [B,H,W] complex
#             # print(f'x_hat = self.ASM_backward(z_new, TF_hat) x_hat.shape:{x_hat.shape}')
#             x_hat_2 = c2two(x_hat)  # [B,2,H,W]
#             # print(f'x_hat_2 = c2two(x_hat) x_hat_2.shape:{x_hat_2.shape}')

#             # 时间步（0..fold_iters-1）
#             # t = torch.full((B,), 1, dtype=torch.long, device=device)
#             x_den_2 = self.denoiser(x_hat_2)  # [B,2,H,W]
#             # print(f'x_den_2.shape:{x_den_2.shape}')
#             x = two2c(x_den_2)  # 回到 complex 物面
#             # print(f'x = two2c(x_den_2) x.shape:{x.shape}')

#         # ---- 输出：最终物面复场 + 像面重建振幅 ----
#         y_rec = torch.abs(self.ASM_forward(x, TF_hat))  # [B,H,W] float

#         return x, y_rec    
    
# class PINet_cpx_uformer_without_timesteps_3layers(nn.Module):
#     """
#     使用 ComplexUformer 作为复数去噪器/修正器。
#     y: 传入的是“衍射振幅” (sensor amplitude)，形状 [B,1,H,W] 或 [B,H,W]
#     TF_hat: 角谱/ASM 的传递函数，复数 [B,H,W] 或 [H,W]（会广播）
#     size: 输入图像边长 H=W=size
#     fold_iters: 折叠次数（迭代步数，建议与 timestep 一起使用）
#     """
#     def __init__(
#         self,
#         size = (128,128),
#         fold_iters: int = 8,
#         u_embed_dim: int = 12,
#         win_size: int = 4
#     ):
#         super().__init__()
#         self.size = size
#         self.fold_iters = fold_iters

#         # ---- 实例化 ComplexUformer（确保 out_channel=2）----
#         self.denoiser = ComplexUformer_without_timesteps_3layers(
#             img_size=size, in_chans=1, dd_in=1,
#             embed_dim=u_embed_dim, win_size=win_size,
#             depths=[1,1,1,1,1,1,1],
#             num_heads=[1,1,1,1,1,1,1],
#             drop_path_rate=0.05,
#             cond_lq=False   # 使用条件 lq（低质量先验）
#         )

#         # 把 ComplexUformer 里用到的时间 MLP 接口绑进来（或直接让其内部调用）
#         self.emb_dim = self.denoiser.embed_dim
#         self.time_embed = self.denoiser.time_embed  # 直接复用其 MLP

#     # ----- ASM 前/后向 -----
#     @staticmethod
#     def ASM_forward(obj, TF):
#         objFT = torch.fft.fftshift(torch.fft.fft2(obj))
#         return torch.fft.ifft2(torch.fft.ifftshift(objFT * TF))

#     @staticmethod
#     def ASM_backward(propfield, TF):
#         objFT = torch.fft.fftshift(torch.fft.fft2(propfield))
#         return torch.fft.ifft2(torch.fft.ifftshift(objFT * torch.conj(TF)))

#     def forward(self, y, TF_hat):

#         B = y.shape[0]
#         H,W = self.size
#         device = y.device
#         assert H % 16 == 0 and W % 16 == 0, "H,W 需要被 16 整除（4 次下采样）。"
#         x = torch.ones(B, 1, H, W, dtype=torch.complex64, device=device)

#         # ---- 物面低质量先验 g0----
#         y_c = y * torch.exp(1j * torch.zeros_like(y))
#         # print(f'y_c.shape:{y_c.shape}')
#         g0_2 = c2two(y_c)
#         # print(f'g0_2.shape:{g0_2.shape}')

#         # ---- 迭代折叠 ----
#         for i in range(self.fold_iters):
#             # 物 -> 像
#             z_hat = self.ASM_forward(x, TF_hat)  # [B,1,H,W] complex
#             z_new = y * torch.exp(1j * torch.angle(z_hat))  # [B,H,W] complex
#             # print(f'z_new = y * torch.exp(1j * torch.angle(z_hat)) z_new.shape:{z_new.shape}')
#             x_hat = self.ASM_backward(z_new, TF_hat)  # [B,H,W] complex
#             # print(f'x_hat = self.ASM_backward(z_new, TF_hat) x_hat.shape:{x_hat.shape}')
#             x_hat_2 = c2two(x_hat)  # [B,2,H,W]
#             # print(f'x_hat_2 = c2two(x_hat) x_hat_2.shape:{x_hat_2.shape}')

#             # 时间步（0..fold_iters-1）
#             # t = torch.full((B,), 1, dtype=torch.long, device=device)
#             x_den_2 = self.denoiser(x_hat_2)  # [B,2,H,W]
#             # print(f'x_den_2.shape:{x_den_2.shape}')
#             x = two2c(x_den_2)  # 回到 complex 物面
#             # print(f'x = two2c(x_den_2) x.shape:{x.shape}')

#         # ---- 输出：最终物面复场 + 像面重建振幅 ----
#         y_rec = torch.abs(self.ASM_forward(x, TF_hat))  # [B,H,W] float

#         return x, y_rec        
    

    
class ChannelAttention_cpx(nn.Module):
    def __init__(self, in_channels, ratio):
        super(ChannelAttention_cpx, self).__init__()
        self.avg_pool = ComplexAvgPool2d((Nx//8,Ny//8))
        self.max_pool = ComplexMaxPool2d((Nx//8,Ny//8))

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
    

# old version
# class SpatialAttention_cpx(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention_cpx, self).__init__()

#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = ComplexConv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = ComplexSigmoid()

#     def forward(self, x):
#         x_amp = torch.abs(x)
#         x_phs = torch.angle(x)

#         avg_out_amp = torch.mean(x_amp, dim=1, keepdim=True)
#         avg_out_phs = torch.mean(x_phs, dim=1, keepdim=True)

#         max_out_amp, _ = torch.max(x_amp, dim=1, keepdim=True)
#         max_out_phs, _ = torch.max(x_phs, dim=1, keepdim=True)
        
#         avg_out_c = avg_out_amp * torch.exp(1j*avg_out_phs)  # [B, 1, H, W]
#         max_out_c = max_out_amp * torch.exp(1j*max_out_phs)  # [B, 1, H, W]

#         # 拼接两个通道组：[avg_real, max_real, avg_imag, max_imag]
#         x_cat = torch.cat([avg_out_c, max_out_c], dim=1) # shape: [B, 2, H, W]
#         x_cat = x_cat.to(x.dtype)  # 🔧 显式转换为 complex64，兼容后续操作
        
#         x_attn = self.conv1(x_cat)  # conv1: nn.Conv2d(4, 1, ...)
#         attn = self.sigmoid(x_attn)
        
#         return x * attn

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

    

    
# class cbam_block(nn.Module):
#     def __init__(self, channel, ratio=8, kernel_size=7):
#         super(cbam_block, self).__init__()
#         self.channelattention = ChannelAttention(channel, ratio=ratio)
#         self.spatialattention = SpatialAttention(kernel_size=kernel_size)

#     def forward(self, x):
#         x = x * self.channelattention(x)
#         x = self.spatialattention(x)

#         return x
    
    
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)      
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)    


# ---- 工具：注册反向钩子，模块级统计 ----
def make_grad_hook(name, *, abs_thresh=1e3, growth=10.0):
    state = {'last_out': None}
    def hook(mod, grad_in, grad_out):
        with torch.no_grad():
            # 取最大范数（可能有多个输入/输出梯度）
            gin  = [g for g in grad_in  if g is not None]
            gout = [g for g in grad_out if g is not None]
            in_n  = max([g.norm().item() for g in gin],  default=0.0)
            out_n = max([g.norm().item() for g in gout], default=0.0)

            trigger = False
            reason = []
            if out_n >= abs_thresh: trigger=True; reason.append(f'out≥{abs_thresh:g}')
            last = state['last_out']
            if last and last>0 and (out_n/(last+1e-12) >= growth):
                trigger=True; reason.append(f'×{growth:g} jump')
            if not gin:  # 有时上游断梯度
                trigger=True; reason.append('no grad_in')

            if trigger:
                print(f'[GRAD] {name}: in={in_n:.2e} out={out_n:.2e} <{"+".join(reason)}>')

            state['last_out'] = out_n
    return hook


# ====== 放在你的文件顶部，工具函数 ======
import time
_STAT = {}
def _stat(name, t, abs_thresh=1e2, growth=5.0):
    if t is None or t.numel()==0: return
    with torch.no_grad():
        mx = float(torch.max(torch.abs(t)).detach().cpu())
        st = _STAT.get(name, {'last': None}); last = st['last']
        trig = (mx >= abs_thresh) or (last and last>0 and mx/(last+1e-12) >= growth) \
               or torch.isinf(t).any().item() or torch.isnan(t).any().item()
        if trig:
            print(f'[STAT] {name}: max|·|={mx:.3e}')
        st['last'] = mx; _STAT[name] = st