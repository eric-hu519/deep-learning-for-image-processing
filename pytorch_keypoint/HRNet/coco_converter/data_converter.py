import numpy as np
import os
import json
from tqdm import tqdm
import argparse
#import nbimporter
from format_conversion import *

parser = argparse.ArgumentParser()

parser.add_argument('--json_path', default=r'annotations',
type=str,
help="input: coco format(json)")

parser.add_argument('--save_path', default=r'datasets/annotations', type=str,
help="specify where to save the output dir of labels")
parser.add_argument('--json_type', default=r'xRayBone_val.json', type=str,
help="import train/val or test .json file?")
parser.add_argument('--save_type',default=r'val',type=str,
                    help="export train/val or test .txt file")
parser.add_argument('--img_folder',default=r'datasets/images',
                    help="specify the folder for images")
parser.add_argument('--rename_img',default=r'True',
                    help="rename the images according to id?")
parser.add_argument('--is_display',default=r'True',help="display the image or not?")
arg = parser.parse_args(args=[])




cwd = os.getcwd()
json_file = arg.json_path# Annotation of COCO Object Instance type
save_type = arg.save_type
save_file = arg.save_path # saved path
rename_flag = bool(arg.rename_img)
display_flag = bool(arg.is_display)
img_folder = arg.img_folder
ana_txt_save_path = os.path.join(cwd,save_file,save_type)
json_type = arg.json_type
json_path = os.path.join(cwd, json_file)
file_path = os.path.join(json_path, json_type)
img_folder = os.path.join(cwd,img_folder,save_type)
conv = my_converter(file_path,save_type,ana_txt_save_path,img_folder,rename_flag,display_flag)
conv.converter()
print("convert result has been saved to " + ana_txt_save_path)

