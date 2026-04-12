from torch import nn
import torch
from torchvision import models, transforms
from utilities import normalize_tensor

class ComplexLoss_re_im(nn.Module):
    def __init__(self, weight_real=1.0, weight_imag=1.0, weight_diff=1.0):
        super(ComplexLoss_re_im, self).__init__()
        self.weight_real = weight_real
        self.weight_imag = weight_imag
        self.weight_diff = weight_diff
        self.mse_loss = nn.MSELoss()  

    def forward(self, pred, target, diff, y_rec):
        # 分别计算幅值和相位
        pred_real = torch.real(pred)  # 预测的幅值
        target_real = torch.real(target)  # 目标的幅值

        pred_imag = torch.imag(pred)  # 预测的相位
        target_imag = torch.imag(target)  # 目标的相位

        # 分别计算L2损失
        real_loss = self.mse_loss(pred_real, target_real)
        imag_loss = self.mse_loss(pred_imag, target_imag)
        diff_loss = self.mse_loss(y_rec, diff)
        # 总损失，加权求和
        total_loss = self.weight_real * real_loss + self.weight_imag * imag_loss + self.weight_diff * diff_loss
        return total_loss
    
    

class ComplexLoss_amp_phs(nn.Module):
    def __init__(self, weight_amp=1.0, weight_phs=1.0, weight_diff=1.0):
        super(ComplexLoss_amp_phs, self).__init__()
        self.weight_amp = weight_amp
        self.weight_phs = weight_phs
        self.weight_diff = weight_diff
        self.mse_loss = nn.MSELoss()  

    def forward(self, pred, target, diff, y_rec):
        # 分别计算幅值和相位
        pred_amp = torch.abs(pred)  # 预测的幅值
        target_amp = torch.abs(target)  # 目标的幅值

        pred_phs = torch.angle(pred)  # 预测的相位
        target_phs = torch.angle(target)  # 目标的相位

        # pred_amp = normalize_tensor(pred_amp)
        # target_amp = normalize_tensor(target_amp)
        # pred_phs = normalize_tensor(pred_phs)
        # target_phs = normalize_tensor(target_phs)
        # diff = normalize_tensor(diff)
        # y_rec = normalize_tensor(y_rec)

        # 分别计算L2损失
        amp_loss = self.mse_loss(pred_amp, target_amp)
        phs_loss = self.mse_loss(pred_phs, target_phs)
        diff_loss = self.mse_loss(y_rec, diff)
        # 总损失，加权求和
        total_loss = self.weight_amp * amp_loss + self.weight_phs * phs_loss + self.weight_diff * diff_loss
        return total_loss
    
class ComplexLoss_amp(nn.Module):
    def __init__(self, weight_amp=1.0, weight_diff=1.0):
        super(ComplexLoss_amp, self).__init__()
        self.weight_amp = weight_amp
        self.weight_diff = weight_diff
        self.mse_loss = nn.MSELoss()  

    def forward(self, pred, target, diff, y_rec):
        # 分别计算幅值和相位
        pred_amp = torch.abs(pred)  # 预测的幅值
        target_amp = torch.abs(target)  # 目标的幅值

        pred_phs = torch.angle(pred)  # 预测的相位
        target_phs = torch.angle(target)  # 目标的相位

        # 分别计算L2损失
        amp_loss = self.mse_loss(pred_amp, target_amp)
        diff_loss = self.mse_loss(y_rec, diff)
        # 总损失，加权求和
        total_loss = self.weight_amp * amp_loss + self.weight_diff * diff_loss
        return total_loss
    

    
class FineTuneLoss(nn.Module):
    def __init__(self, diffLoss_weight=1):
        super(FineTuneLoss, self).__init__()
        self.mse_loss = nn.MSELoss() 
        self.diffLoss_weight = diffLoss_weight 

    def forward(self, diff, y_rec):
        diff_loss = self.mse_loss(y_rec, diff) * self.diffLoss_weight
        return diff_loss
    

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
