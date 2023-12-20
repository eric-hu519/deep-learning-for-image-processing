
import numpy as np
import cv2
import torch.utils.data as data
from sklearn.model_selection import KFold
import copy
import torch
import numpy as np
import cv2
import torch.utils.data as data
from pycocotools.coco import COCO
import os

class MyDataset(data.Dataset):
    def __init__(self,root,data, transform=None):
        super().__init__()
        self.data = data
        self.transform = transform#预处理函数
        self.anno_path = os.path.join(root, "annotations", "allanno","allanno.json")
        self.coco = COCO(self.anno_path)
        
    def __getitem__(self, idx):
        target = copy.deepcopy(self.data[idx])
        #print(target[1]["image_path"])
        image = cv2.imread(target[1]["image_path"])
        if self.transform is not None:
            image, person_info = self.transform(image, target[1])
        return image, target[1]
    
    @staticmethod
    def collate_fn(batch):
        imgs_tuple, targets_tuple = tuple(zip(*batch))
        imgs_tensor = torch.stack(imgs_tuple)
        return imgs_tensor, targets_tuple


    def __len__(self):
        return len(self.data)