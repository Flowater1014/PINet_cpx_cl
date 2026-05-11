from torch import nn
import torch

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
    
