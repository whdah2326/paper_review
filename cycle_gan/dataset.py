import os
import glob
import torch
import cv2
import random
import albumentations as A

from torch. utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from typing import List, Dict



class UnalignedDataset(Dataset):
    def __init__(self, data_root_A: str, data_root_B: str, is_train:bool=True, transforms=ToTensorV2()):
        super(UnalignedDataset, self).__init__()
        self.data_root_A = data_root_A
        self.data_root_B = data_root_B
        self.is_train = is_train
        self.transforms = transforms

        paths_A = sorted(self._load_image_path(self.data_root_A))
        paths_B = sorted(self._load_image_path(self.data_root_B))

        self.image_paths_A, self.image_paths_B = self._adjust_dataset_length(paths_A, paths_B)

    
    def __len__(self):
        return len(self.image_paths_A)

    
    def __getitem__(self, index:int)->Dict:
        A = cv2.cvtColor(cv2.imread(self.image_paths_A[index]), cv2.COLOR_BGR2RGB)
        B = cv2.cvtColor(cv2.imread(self.image_paths_B[index]), cv2.COLOR_BGR2RGB)

        if self.transforms:
            A = self.transforms(image=A)['image']
            B = self.transforms(image=B)['image']

        return {'A': A, 'B': B}

    
    def _load_image_path(self, data_dir:str)->List[str]:
        image_path = glob.glob(data_dir+"/*")

        return image_path
    

    def _adjust_dataset_length(self, paths_A:str, paths_B:str):
        min_len = min(len(paths_A), len(paths_B))
        
        return paths_A[:min_len], paths_B[:min_len]
    

if __name__ == "__main__":
    img_dir = "...insert path"
    img_dir = glob.glob(img_dir+"/*")
    img = cv2.imread(img_dir[2])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)