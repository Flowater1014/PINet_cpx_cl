from torch.utils.data import Dataset
import torch
from utilities import *
from PIL import Image
import os
import numpy as np

class ComplexDataset(Dataset):
    def __init__(self, amp_folder, phs_folder, num=2000, transform=None, batch_size = 8):
        self.amp_folder = amp_folder  
        self.phs_folder = phs_folder  
        self.transform = transform  
        self.batch_size = batch_size
        self.rng_distance = np.random.default_rng(seed=1)
        self.rng_phs_1 = np.random.default_rng(seed=2)
        self.rng_phs_2 = np.random.default_rng(seed=3)

        num_batches = (num + batch_size - 1) // batch_size  # 向上取整
        self.distance_batch = self.rng_distance.uniform(2e-3, 3e-3, num_batches)
        
        
        # 获取文件名列表，文件名为 1.png, 2.png, ..., 1000.png
        self.files = [f"{i}.png" for i in range(1, num+1)]

    def __len__(self):
        return len(self.files)
    

    def __getitem__(self, idx):

        batch_id = idx // self.batch_size
        z = self.distance_batch[batch_id]

        # 固定距离
        TF = np.exp(1j*z*np.sqrt(k**2-(2*np.pi*FX)**2-(2*np.pi*FY)**2))
        TF_torch = torch.from_numpy(TF).to(device).to(torch.complex64)

        
        # 获取文件名
        img_name = self.files[idx]
        img_amp_path = os.path.join(self.amp_folder, img_name)
        img_phs_path = os.path.join(self.phs_folder, img_name)
        img_amp = Image.open(img_amp_path)
        img_phs = Image.open(img_phs_path)

        if self.transform:
            img_amp_tensor = self.transform(img_amp)
            img_phs_tensor = self.transform(img_phs)

        x = self.rng_phs_1.uniform(-0.5, 0.5)

        # img_phs_tensor = 2*np.pi*img_phs_tensor - np.pi
        img_phs_tensor = img_phs_tensor - 0.5
        # img_phs_tensor = torch.clip(img_phs_tensor,-1,1)
            
        # 数据集还取幅值和相位
        label_tensor = img_amp_tensor * torch.exp(1j * img_phs_tensor)
        diff_tensor = diff_compute(label_tensor,TF_torch).float()

        return diff_tensor,label_tensor,TF_torch

class ComplexDataset_npy(Dataset):
    def __init__(self, data_folder, num=1000, batch_size = 4):
        self.amp_path = os.path.join(data_folder,'label_amp.npy')
        self.phs_path = os.path.join(data_folder,'label_phs.npy')
        self.amp = np.load(self.amp_path)
        self.phs = np.load(self.phs_path)
        self.batch_size = batch_size
        self.rng_distance = np.random.default_rng(seed=1)
        num_batches = (num + batch_size - 1) // batch_size  # 向上取整
        self.distance_batch = self.rng_distance.uniform(2e-3, 3e-3, num_batches)

    def __len__(self):
        return len(self.amp)
    

    def __getitem__(self, idx):

        batch_id = idx // self.batch_size
        z = self.distance_batch[batch_id]

        # 固定距离
        TF = np.exp(1j*z*np.sqrt(k**2-(2*np.pi*FX)**2-(2*np.pi*FY)**2))
        TF_torch = torch.from_numpy(TF).to(device).to(torch.complex64)

        
        # 获取文件名
        amp_np = self.amp[idx]
        phs_np = self.phs[idx]

        amp_tensor = torch.from_numpy(amp_np)
        phs_tensor = torch.from_numpy(phs_np)

        if amp_tensor.ndim == 2:
            amp_tensor = amp_tensor.unsqueeze(0)
            phs_tensor = phs_tensor.unsqueeze(0)

        # 数据集还取幅值和相位
        label_tensor = amp_tensor * torch.exp(1j * phs_tensor)
        diff_tensor = diff_compute(label_tensor,TF_torch).float()

        return diff_tensor,label_tensor,TF_torch

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
    
class AmpDataset(Dataset):
    def __init__(self, amp_folder, num=1000, transform=None):
        self.amp_folder = amp_folder  
        self.transform = transform  
        
        # 获取文件名列表，文件名为 1.png, 2.png, ..., 1000.png
        self.files = [f"{i}.png" for i in range(1, num+1)]

    def __len__(self):
        return len(self.files)
    

    def __getitem__(self, idx):
        # 获取文件名
        img_name = self.files[idx]
        img_amp_path = os.path.join(self.amp_folder, img_name)
        img_amp = Image.open(img_amp_path)

        if self.transform:
            img_amp_tensor = self.transform(img_amp)
        # 数据集还取幅值和相位
        label_tensor = img_amp_tensor
        diff_tensor = diff_compute(label_tensor).float()

        return diff_tensor,label_tensor
    
class DenoiseDataset(Dataset):
    def __init__(self, LQ_folder, GT_folder, num=1000, transform=None):
        self.LQ_folder = LQ_folder  
        self.GT_folder = GT_folder  
        self.transform = transform  
        
        # 获取文件名列表，文件名为 1.png, 2.png, ..., 1000.png
        self.files = [f"{i}.png" for i in range(1, num+1)]
        

    def __len__(self):
        return len(self.files)
    

    def __getitem__(self, idx):
        # 获取文件名
        img_name = self.files[idx]
        img_LQ_path = os.path.join(self.LQ_folder, img_name)
        img_GT_path = os.path.join(self.GT_folder, img_name)
        img_LQ = Image.open(img_LQ_path)
        img_GT = Image.open(img_GT_path)

        if self.transform:
            img_LQ_tensor = self.transform(img_LQ)
            img_GT_tensor = self.transform(img_GT)

        return img_LQ_tensor,img_GT_tensor
    
class ComplexDenoiseDataset(Dataset):
    def __init__(self, LQ_folder_amp,LQ_folder_phs,GT_folder_amp,GT_folder_phs, num=1000, transform=None):
        self.LQ_folder_amp = LQ_folder_amp  
        self.GT_folder_amp = GT_folder_amp  
        self.LQ_folder_phs = LQ_folder_phs  
        self.GT_folder_phs = GT_folder_phs 
        self.transform = transform  
        
        # 获取文件名列表，文件名为 1.png, 2.png, ..., 1000.png
        self.files = [f"{i}.png" for i in range(1, num+1)]

    def __len__(self):
        return len(self.files)
    

    def __getitem__(self, idx):
        # 获取文件名
        img_name = self.files[idx]
        img_LQ_amp_path = os.path.join(self.LQ_folder_amp, img_name)
        img_GT_amp_path = os.path.join(self.GT_folder_amp, img_name)
        img_LQ_amp = Image.open(img_LQ_amp_path)
        img_GT_amp = Image.open(img_GT_amp_path)
        
        img_LQ_phs_path = os.path.join(self.LQ_folder_phs, img_name)
        img_GT_phs_path = os.path.join(self.GT_folder_phs, img_name)
        img_LQ_phs = Image.open(img_LQ_phs_path)
        img_GT_phs = Image.open(img_GT_phs_path)

        if self.transform:
            img_LQ_amp_tensor = self.transform(img_LQ_amp)
            img_GT_amp_tensor = self.transform(img_GT_amp)
            img_LQ_phs_tensor = self.transform(img_LQ_phs)
            img_GT_phs_tensor = self.transform(img_GT_phs)
            
        img_LQ_cpl_tensor = img_LQ_amp_tensor * torch.exp(1j*img_LQ_phs_tensor)
        img_GT_cpl_tensor = img_GT_amp_tensor * torch.exp(1j*img_GT_phs_tensor)        
            
        return img_LQ_cpl_tensor,img_GT_cpl_tensor