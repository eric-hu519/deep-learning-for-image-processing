import numpy as np
import os
import json
from tqdm import tqdm

class my_converter:
    def __init__(self,file_path,save_type,ana_txt_save_path,img_folder,rename_flag,display_flag = False):
        self.file_path = file_path
        self.save_type = save_type
        self.ana_txt_save_path = ana_txt_save_path
        self.img_folder = img_folder
        self.rename_flag = rename_flag
        self.display_flag = display_flag
    def converter(self):
        data = json.load(open(self.file_path, 'r'))
        if not os.path.exists(self.ana_txt_save_path):
            os.makedirs(self.ana_txt_save_path)

        #list_file = open(os.path.join(self.ana_txt_save_path, self.save_type+'.txt'), 'w')
        
        anno_count = 0 #count for same image annotations
        last_img_id = -1 #init img counting, make sure entering the loop
        accuracy = 10 ** 6 
        final_annotations = {}
        final_category = {}
        #rename images by image_id
        if self.rename_flag:
            data = self.rename_img(data)

        #get the first category in data['categories']
        category = data['categories'][0]
        #change category
        category['name'] = 'spineopelvic'
        category['id'] = 1
        category['keypoints'] = ["pelvis","l_hip","r_hip","spine"]
        final_category[category['id']] = category


        for annotations in tqdm(data['annotations']):
        # filename = annotations["file_name"]
            #if  annotations["keypoints"] != None:
            if last_img_id != annotations["image_id"]:#if new image
                if anno_count == 0:
                    bbox = [annotations["bbox"]]
                    bbox_center = self.bbox_converter(annotations["bbox"])
                    new_keypoints = bbox_center
                    keypoints = annotations['keypoints']
                    keypoints.extend(new_keypoints)#extend keypoints
                    anno_count += 1
                elif anno_count == 1:
                    bbox.extend([annotations["bbox"]]) # extend bbox
                    bbox_center = self.bbox_converter(annotations["bbox"])
                    new_keypoints = bbox_center
                    keypoints.extend(new_keypoints)
                    anno_count += 1
                elif anno_count == 2 :
                    bbox.extend([annotations["bbox"]])
                    bbox_center = self.bbox_converter(annotations["bbox"])
                    new_keypoints = bbox_center
                    keypoints.extend(new_keypoints)
                    #rewrite annotations
                    annotations['keypoints'] = keypoints
                    annotations['bbox'] = self.merge_bbox(bbox)
                    annotations['num_keypoints'] = 4
                    annotations['category_id'] = 1
                    annotations['id'] = annotations["image_id"]
                    annotations['area'] = annotations['bbox'][2]*annotations['bbox'][3]
                    final_annotations[annotations["image_id"]] = annotations
                    
                    anno_count = 0#reset count

        final_annotations = list(final_annotations.values())
        data['annotations'] = final_annotations
        data['categories'] = list(final_category.values())
        with open(os.path.join(self.ana_txt_save_path,self.save_type+"_converted"".json"),'w') as f:
            json.dump(data,f)
                    
        if self.display_flag:
            self.annocheck(data)
        print("Done!\n")

        
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
    def merge_bbox(self,bboxes):
        min_x = min(bbox[0] for bbox in bboxes)
        min_y = min(bbox[1] for bbox in bboxes)
        max_x = max(bbox[0] + bbox[2] for bbox in bboxes)
        max_y = max(bbox[1] + bbox[3] for bbox in bboxes)

        merged_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
        return merged_bbox
    #randomly choose a image to display the annotations
    def annocheck(self, data):
        import cv2
        import random
        import matplotlib.pyplot as plt

        img = random.choice(data['images'])
        img_id = img["id"]
        img_name = img["file_name"]
        img_path = os.path.join(self.save_type, self.img_folder, img_name)
        img = cv2.imread(img_path)

        colors = [(0, 253, 79), (255, 0, 0), (0, 0, 255), (255, 255, 79)]  

        for anno in data['annotations']:
            if anno['image_id'] == img_id:
                bbox = anno['bbox']
                keypoints = anno['keypoints']

                for i in range(0, len(keypoints), 3):
                    x = int(keypoints[i])
                    y = int(keypoints[i+1])
                    visibility = keypoints[i+2]

                    if visibility == 2:
                        # Keypoint is visible
                        color = colors[i // 3 % len(colors)]
                        cv2.circle(img, (x, y), 3, color, -1)

                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (255, 255, 0), 2)

        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()