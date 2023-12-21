
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
    def __init__(self,root, data, transform=None,train_ids=None):
        super().__init__()
        self.data = data
        self.transform = transform#预处理函数
        self.anno_path = os.path.join(root, "annotations", "allanno","allanno.json")
        self.coco = COCO(self.anno_path)

        # 获取训练集或测试集的图像ID
        #+1是因为coco数据集的图片id是从1开始的
        self.train_ids = [i+1 for i in train_ids]

        #过滤数据标签
        self.coco.anns = {ann_id: ann for ann_id, ann in self.coco.anns.items() if ann['image_id'] in self.train_ids}
        self.coco.imgs = {img_id: img for img_id, img in self.coco.imgs.items() if img_id in self.train_ids}
        #self.coco = self.coco
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