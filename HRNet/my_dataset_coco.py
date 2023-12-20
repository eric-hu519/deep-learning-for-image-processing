import os
import copy

import torch
import numpy as np
import cv2
import torch.utils.data as data
from pycocotools.coco import COCO


class CocoKeypoint(data.Dataset):
    def __init__(self,
                 root,#指向coco2017文件夹路径
                 dataset="train",#取train or validation两个字符串，指定训练集还是验证集
                 years="2017",
                 transforms=None,#是否进行预处理
                 det_json_path=None,#None则使用coco的GT信息
                 fixed_size=(256, 192)):
        super().__init__()
        
        assert dataset in ["train", "val", "test","allanno"], 'dataset must be in ["train", "val", "test","allanno"]'
        if dataset == "allanno":
            anno_file = f"{dataset}.json"
        else:
            anno_file = f"{dataset}_converted.json"
        assert os.path.exists(root), "file '{}' does not exist.".format(root)
        self.img_root = os.path.join(root,"images", f"{dataset}")
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
        self.anno_path = os.path.join(root, "annotations", f"{dataset}",anno_file)
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)

        self.fixed_size = fixed_size
        self.mode = dataset
        self.transforms = transforms
        self.coco = COCO(self.anno_path)
        img_ids = list(sorted(self.coco.imgs.keys()))

        if det_json_path is not None:
            det = self.coco.loadRes(det_json_path)
        else:
            det = self.coco

        self.valid_person_list = []
        #初始化目标索引
        obj_idx = 0
        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            #通过图像id找到对应标注id
            ann_ids = det.getAnnIds(imgIds=img_id)
            anns = det.loadAnns(ann_ids)
            for ann in anns:
                # only save person class
                if ann["category_id"] != 1:#类别id不为1，跳过
                    print(f'warning: find not support id: {ann["category_id"]}, only support id: 1 (person)')
                    continue

                # COCO_val2017_detections_AP_H_56_person.json文件中只有det信息，没有keypoint信息，跳过检查
                if det_json_path is None:
                    # skip objs without keypoints annotation
                    if "keypoints" not in ann:
                        continue
                    if max(ann["keypoints"]) == 0:
                        continue

                xmin, ymin, w, h = ann['bbox']#获取对应bounding box的坐标
                # Use only valid bounding boxes
                if w > 0 and h > 0:
                    info = {
                        "box": [xmin, ymin, w, h],
                        "image_path": os.path.join(self.img_root, img_info["file_name"]),
                        "image_id": img_id,
                        "image_width": img_info['width'],
                        "image_height": img_info['height'],
                        "obj_origin_hw": [h, w],
                        "obj_index": obj_idx,
                        "score": ann["score"] if "score" in ann else 1.
                    }

                    # COCO_val2017_detections_AP_H_56_person.json文件中只有det信息，没有keypoint信息，跳过
                    if det_json_path is None:
                        keypoints = np.array(ann["keypoints"]).reshape([-1, 3])
                        #提取各关键点可见度信息
                        visible = keypoints[:, 2]
                        keypoints = keypoints[:, :2]
                        info["keypoints"] = keypoints
                        
                        info["visible"] = visible

                    self.valid_person_list.append(info)
                    obj_idx += 1

    def __getitem__(self, idx):
        target = copy.deepcopy(self.valid_person_list[idx])
        image = cv2.imread(target["image_path"])
        #转换颜色格式
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            #进行数据增强
            image, person_info = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.valid_person_list)



if __name__ == '__main__':
    train = CocoKeypoint("datasets", dataset="train")#输入参数
    print(len(train))
    
    
