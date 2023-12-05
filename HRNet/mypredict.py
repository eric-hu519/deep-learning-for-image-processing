import json
import os
import datetime
from pathlib import Path
import torch
from torch.utils import data
import numpy as np
import re
import glob
import transforms
from model import HighResolutionNet
from my_dataset_coco import CocoKeypoint
from train_utils import train_eval_utils as utils


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path



def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    #create model
    model = HighResolutionNet(base_channel=32)
    weights = torch.load("./save_weights/exp134/best_model-127.pth", map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    model.load_state_dict(weights)



    with open(args.keypoints_path, "r") as f:
        person_kps_info = json.load(f)
    #get image size info
    fixed_size = args.fixed_size
    #get heatmap size info(which is 1/4 of image size) 
    heatmap_hw = (args.fixed_size[0] // 4, args.fixed_size[1] // 4)
    kps_weights = np.array(person_kps_info["kps_weights"],
                           dtype=np.float32).reshape((args.num_joints,))
    data_transform = {
        "train": transforms.Compose([
            #randomly crop a person from the image
            #according to upper body index and lower body index in annotation
            #transforms.HalfBody(0.3, person_kps_info["upper_body_ids"], person_kps_info["lower_body_ids"]),
            #randomly resize and rotate input image
            transforms.AffineTransform(scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=fixed_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomContrast(0.5),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.AffineTransform(scale=(1.00, 1), fixed_size=fixed_size),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    data_root = args.data_path

    # load train data set
    # coco2017 -> annotations -> person_keypoints_train2017.json

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    #instance dataset
    # load validation data set
    # coco2017 -> annotations -> person_keypoints_val2017.json
    val_dataset = CocoKeypoint(data_root, "test", transforms=data_transform["val"], fixed_size=args.fixed_size,
                               det_json_path=args.person_det)
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=nw,
                                      collate_fn=val_dataset.collate_fn)

    # print(model)

    model.to(device)

    # define optimizer

    val_map = []
    sc_abs_error = []
    s1_abs_error = []
    fh1_abs_error = []
    fh2_abs_error = []
    val_loss = []
    best_err = np.zeros((4,))

    


    # evaluate on the test dataset
    coco_info = utils.evaluate(model, val_data_loader, device=device,
                            flip=True)


    #检查runs文件夹是否存在，若不存在则创建
    #if not os.path.exists("./runs"):
    #    os.makedirs("./runs")
    # 将实验结果写入txt，保存在runs文件夹下
    val_map.append(coco_info[1])  # @0.5 mAP
    s1_abs_error.append(coco_info[10])
    sc_abs_error.append(coco_info[11])
    fh1_abs_error.append(coco_info[12])
    fh2_abs_error.append(coco_info[13])

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map,str(args.log_path))
    #plot abs error curve
    if len(sc_abs_error) != 0:
        from plot_curve import plot_abs_error
        plot_abs_error(sc_abs_error,'sc',str(args.log_path))
        plot_abs_error(s1_abs_error,'s1',str(args.log_path))
        plot_abs_error(fh1_abs_error,'fh1',str(args.log_path))
        plot_abs_error(fh2_abs_error,'fh2',str(args.log_path))
    if len(best_err) != 0:
        print("min_sc_e: ",best_err[0],"\n",
              "min_s1_e: ",best_err[1],"\n",
              "min_fh1_e: ",best_err[2],"\n",
              "min_fh2_e: ",best_err[3],"\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录(coco2017)
    parser.add_argument('--data-path', default='datasets', help='dataset')
    # COCO数据集人体关键点信息
    parser.add_argument('--keypoints-path', default="./spinopelvic_keypoints.json", type=str,
                        help='person_keypoints.json path')
    # 原项目提供的验证集person检测信息，如果要使用GT信息，直接将该参数置为None，建议设置成None
    parser.add_argument('--person-det', type=str, default=None)
    parser.add_argument('--fixed-size', default=[512,512], nargs='+', type=int, help='input size')
    # keypoints点数
    parser.add_argument('--num-joints', default=4, type=int, help='num_joints')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights/exp', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[150, 200], nargs='+', type=int, help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.3258, type=float, help='decrease lr by a factor of lr-gamma')
    # 学习率
    parser.add_argument('--lr', default=0.00188, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # AdamW的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1.87e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 训练的batch size
    parser.add_argument('--batch-size', default=28, type=int, metavar='N',
                        help='batch size when training.')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--savebest", default = 1, help="save best model")
    parser.add_argument("--log-path", default = "./tests/exp", help="log path")
    parser.add_argument("--with_FFCA", default= True , help="enable FFCA")
    parser.add_argument("--optimizer", default="adamw", help="optimizer")
    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    args.output_dir = increment_path(Path(args.output_dir), exist_ok=False,mkdir=True)
    args.log_path = increment_path(Path(args.log_path), exist_ok=False,mkdir=True)
    #args.fixed_size = [args.fixed_size[0], args.fixed_size[0]]
    #steps = args.lr_steps[0]
    #next_steps = steps + 50
    #args.lr_steps = [steps, next_steps]
    main(args)
