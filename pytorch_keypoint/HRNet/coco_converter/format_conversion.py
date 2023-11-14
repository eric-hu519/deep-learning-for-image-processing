import numpy as np
import os
import json
from tqdm import tqdm

class my_converter:
    def __init__(self,file_path,save_type,ana_txt_save_path,img_folder,rename_flag):
        self.file_path = file_path
        self.save_type = save_type
        self.ana_txt_save_path = ana_txt_save_path
        self.img_folder = img_folder
        self.rename_flag = rename_flag
    def converter(self):
        data = json.load(open(self.file_path, 'r'))
        if not os.path.exists(self.ana_txt_save_path):
            os.makedirs(self.ana_txt_save_path)

        #list_file = open(os.path.join(self.ana_txt_save_path, self.save_type+'.txt'), 'w')
        
        anno_count = 0 #count for same image annotations
        last_img_id = -1 #init img counting, make sure entering the loop
        accuracy = 10 ** 6 
        keypoints = [] #init key points lists
        lables = []
        #rename images by image_id
        if self.rename_flag:
            data = self.rename_img(data)

        for annotations in tqdm(data['annotations']):
        # filename = annotations["file_name"]
            #if  annotations["keypoints"] != None:
            if anno_count == 0:
                bbox = annotations["bbox"]
                bbox_center = self.bbox_converter(bbox)
                new_keypoints = bbox_center
                annotations['keypoints'].extend(new_keypoints)
                annotations['num_keypoints'] = 2
                anno_count += 1
            elif anno_count == 1:
                bbox = annotations["bbox"]
                bbox_center = self.bbox_converter(bbox)
                new_keypoints = bbox_center
                if 'keypoints' not in annotations:
                    annotations['keypoints'] = []
                    annotations['num_keypoints'] = 1
                    annotations['keypoints'].extend(new_keypoints)
                    anno_count += 1
            elif anno_count == 2 :
                bbox = annotations["bbox"]
                bbox_center = self.bbox_converter(bbox)
                new_keypoints = bbox_center
                if 'keypoints' not in annotations:
                    annotations['keypoints'] = []
                    annotations['num_keypoints'] = 1
                    annotations['keypoints'].extend(new_keypoints)
                anno_count = 0#reset count
        with open(os.path.join(self.ana_txt_save_path,self.save_type+"_converted"".json"),'w') as f_txt:
            json.dump(data,f_txt)
                    
                

        
    def bbox_converter(self,bbox):
            x_min, y_min, bb_width, bb_height = bbox
            x_max = x_min + bb_width
            y_max = y_min + bb_height
            # compute center coordinates
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            bbox_center = [center_x, center_y, 2]
            return bbox_center

    def val_norm(self,value,img_width,img_hight,accuracy):
            value[0] = int((value[0] / img_width) * accuracy) / accuracy
            value[1] = int((value[1] / img_hight) * accuracy) / accuracy
            return value
    
    def rename_img(self,data):
        for img in tqdm(data['images']):
                img_id = img["id"]
                img["file_name"] = str(img_id)+".jpg"
        return data
