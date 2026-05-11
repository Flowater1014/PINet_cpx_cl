from torch.utils.data import Dataset
import torch
from src.utilities import *
from PIL import Image
import os
import numpy as np

# 对ComplexDataset_npy的改良，把不该出现在dataset的传播距离剥离
class ComplexDataset_npy_plus(Dataset):
    def __init__(self, data_folder, num=1000):
        self.amp_path = os.path.join(data_folder,'label_amp.npy')
        self.phs_path = os.path.join(data_folder,'label_phs.npy')
        self.amp = np.load(self.amp_path)
        self.phs = np.load(self.phs_path)

    def __len__(self):
        return len(self.amp)

    def __getitem__(self, idx):
        # 获取文件名
        amp_np = self.amp[idx]
        phs_np = self.phs[idx]

        amp_tensor = torch.from_numpy(amp_np)
        phs_tensor = torch.from_numpy(phs_np)

        if amp_tensor.ndim == 2:
            amp_tensor = amp_tensor.unsqueeze(0)
        if phs_tensor.ndim == 2:        
            phs_tensor = phs_tensor.unsqueeze(0)
            
        label_tensor = amp_tensor * torch.exp(1j * phs_tensor)
        return label_tensor

# 对ComplexDataset的改良，把不该出现在dataset的传播距离剥离
class ComplexDataset_png_plus(Dataset):
    def __init__(self, data_folder, num=1000, transform=None):
        self.amp_path = os.path.join(data_folder,'amp')
        self.phs_path = os.path.join(data_folder,'phs')
        # 获取文件名列表，文件名为 1.png, 2.png, ..., 1000.png
        self.files = [f"{i}.png" for i in range(1, num+1)]
        self.transform = transform
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_amp_path = os.path.join(self.amp_path, img_name)
        img_phs_path = os.path.join(self.phs_path, img_name)
        img_amp = Image.open(img_amp_path)
        img_phs = Image.open(img_phs_path)

        if self.transform:
            img_amp_tensor = self.transform(img_amp)
            img_phs_tensor = self.transform(img_phs)

        img_phs_tensor = img_phs_tensor - 0.5
        label_tensor = img_amp_tensor * torch.exp(1j * img_phs_tensor)
        
        return label_tensor

def make_collate_fn(z_low=1.5, z_high=3, k=None, FX=None, FY=None, device='cpu', diff_compute=None):
    def collate_fn(batch):
        label = torch.stack(batch)
        z = np.random.uniform(z_low*1e-3, z_high*1e-3)
        TF = np.exp(1j * z * np.sqrt(k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2))
        TF_torch = torch.from_numpy(TF).to(device).to(torch.complex64)
        diff = diff_compute(label, TF_torch).float()
        return diff, label, TF_torch
    return collate_fn
    
