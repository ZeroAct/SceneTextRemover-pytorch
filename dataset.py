import os, cv2
import numpy as np

import torch
from torch.utils.data import Dataset

def mat_to_tensor(mat):
    mat = mat.transpose((2, 0, 1))
    tensor = torch.Tensor(mat)
    return tensor

def tensor_to_mat(tensor):
    mat = tensor.detach().cpu().numpy()
    mat = mat.transpose((0, 2, 3, 1))
    return mat

def preprocess_image(img, target_shape: tuple):
    img = cv2.resize(img, target_shape, interpolation=cv2.INTER_CUBIC).astype(np.float32)
    img = img / 255.
    if len(img.shape) == 2:
        img = img.reshape(*img.shape, 1)
    
    return img

def postprocess_image(img):
    img = img * 255
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

class CustomDataset(Dataset):
    def __init__(self,
                 data_dir,
                 set_name="train",
                 target_size=(256, 256)):
        
        super().__init__()
        
        self.root_dir = os.path.join(data_dir, set_name)
        self.target_size = target_size
        
        self.I_dir = os.path.join(self.root_dir, "I")
        self.Itegt_dir = os.path.join(self.root_dir, "Itegt")
        self.Mm_dir = os.path.join(self.root_dir, "Mm")
        self.Msgt_dir = os.path.join(self.root_dir, "Msgt")
        
        self.datas = os.listdir(self.I_dir)
        
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        img_name = self.datas[idx]
        
        I      = cv2.imread(os.path.join(self.I_dir, img_name))
        Itegt  = cv2.imread(os.path.join(self.Itegt_dir, img_name))
        Mm     = cv2.imread(os.path.join(self.Mm_dir, img_name), cv2.IMREAD_GRAYSCALE)
        Msgt   = cv2.imread(os.path.join(self.Msgt_dir, img_name), cv2.IMREAD_GRAYSCALE)
        
        I      = mat_to_tensor(preprocess_image(I,     self.target_size))
        Itegt  = mat_to_tensor(preprocess_image(Itegt, self.target_size))
        Mm     = mat_to_tensor(preprocess_image(Mm,    self.target_size))
        Msgt   = mat_to_tensor(preprocess_image(Msgt,  self.target_size))
        
        return I, Itegt, Mm, Msgt
        

if __name__ == "__main__":
    ds = CustomDataset('dataset', 'train')
    
    I, Itegt, Mm, Ms = ds.__getitem__(0)
    print(f"Dataset length : {len(ds)}")
    print(f"I shape : {I.shape}")
    print(f"Itegt shape : {Itegt.shape}")
    print(f"Mm shape : {Mm.shape}")
    print(f"Ms shape : {Ms.shape}")
