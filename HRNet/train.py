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
import wandb

def wandb_init(args):
    wandb.init(project="spinopelvic", config=args)
    wandb.run.name = str(args.log_path).split('/')[-1]
    #wandb.run.save()
    wandb.watch_called = False

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config          # Initialize config
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.weight_decay = args.weight_decay
    config.lr_steps = args.lr_steps
    config.lr_gamma = args.lr_gamma
    config.num_joints = args.num_joints
    config.fixed_size = args.fixed_size
    config.with_FFCA = args.with_FFCA

def wandb_log(epoch, train_loss, cocoinfo,best_results):
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": cocoinfo[14],
        "sc_abs_error": cocoinfo[10],
        "s1_abs_error": cocoinfo[11],
        "fh1_abs_error": cocoinfo[12],
        "fh2_abs_error": cocoinfo[13],
        "sc_best": best_results[0],
        "s1_best": best_results[1],
        "fh1_best": best_results[2],
        "fh2_best": best_results[3],

        "Avg Precision @[ IoU=0.50:0.95 | area=   all | maxDets=20 ]": cocoinfo[0],
        "Avg Precision @[ IoU=0.50      | area=   all | maxDets=20 ]": cocoinfo[1],
        "Avg Precision @[ IoU=0.75      | area=   all | maxDets=20 ]": cocoinfo[2],
        "Avg Precision @[ IoU=0.50:0.95 | area= large | maxDets=20 ]": cocoinfo[4],
        "Avg Recall @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]": cocoinfo[5],
        "Avg Recall @[ IoU=0.50 | area= all | maxDets= 20 ]": cocoinfo[6],
        "Avg Recall @[ IoU=0.75 | area= all | maxDets= 20 ]": cocoinfo[7],
        "Avg Recall @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]": cocoinfo[9],
    })

def create_model(num_joints, load_pretrain_weights=True, with_FFCA=True):
    #base_channel=32 means HRnet-w32
    model = HighResolutionNet(base_channel=32, num_joints=num_joints, with_FFCA=with_FFCA)
    
    if load_pretrain_weights:
        # 载入预训练模型权重
        # 链接:https://pan.baidu.com/s/1Lu6mMAWfm_8GGykttFMpVw 提取码:f43o
        weights_dict = torch.load("./hrnet_w32.pth", map_location='cpu')

        for k in list(weights_dict.keys()):
            # 如果载入的是imagenet权重，就删除无用权重，关键点检测不需要全连接层
            if ("head" in k) or ("fc" in k):
                del weights_dict[k]
            if with_FFCA:
                if("stage4.2" in k):
                    del weights_dict[k]

            #如果载入的是coco权重，对比下num_joints，如果不相等就删除
            #if "final_layer" in k:
                #if weights_dict[k].shape[0] != num_joints:
                    #del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0:
            print("missing_keys: ", missing_keys, "\n", "unexpected_keys: ", unexpected_keys)
            # 添加缺失的权重和偏置
            #for key in missing_keys:
                #if 'weight' in key:
                    #torch.nn.init.kaiming_normal_(model.state_dict()[key], mode='fan_out', nonlinearity='relu')
                #elif 'bias' in key:
                    #model.state_dict()[key].zero_()
        

    return model

#check loss list to save the best model
def check_loss_list(loss_list, loss):
    if len(loss_list) == (0 or 1):
        return True
    elif loss <= min(loss_list):
            return True
    else:
        return False

def increment_path(path,refer_path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    refer_path = Path(refer_path)
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        ref_matches = [re.search(rf"%s{sep}(\d+)" % refer_path.stem, d) for d in dirs]
        ref_i = [int(m.groups()[0]) for m in ref_matches if m]  # indices
        ref_n = max(ref_i) + 1 if ref_i else 2  # increment number
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        n = max(n, ref_n)#keep the weight file number same as the log file number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wandb.login()
    wandb_init(args)




    # 用来保存coco_info的文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

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
    train_dataset = CocoKeypoint(data_root, "train", transforms=data_transform["train"], fixed_size=args.fixed_size)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=nw,
                                        collate_fn=train_dataset.collate_fn)
    #instance dataset
    # load validation data set
    # coco2017 -> annotations -> person_keypoints_val2017.json
    val_dataset = CocoKeypoint(data_root, "val", transforms=data_transform["val"], fixed_size=args.fixed_size,
                               det_json_path=args.person_det)
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=nw,
                                      collate_fn=val_dataset.collate_fn)

    # create model
    model = create_model(num_joints=args.num_joints, with_FFCA=args.with_FFCA)
    # print(model)

    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    #define adam as optimizer
    optimizer = torch.optim.AdamW(params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []
    sc_abs_error = []
    s1_abs_error = []
    fh1_abs_error = []
    fh2_abs_error = []
    val_loss = []
    best_err = np.zeros((4,))

    


    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 50 iterations
        #利用均方误差作为损失函数
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True,
                                              scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        mloss = utils.eval_loss(model, val_data_loader, device=device, epoch=epoch, scaler=scaler)

        # evaluate on the test dataset
        coco_info = utils.evaluate(model, val_data_loader, device=device,
                                   flip=True)
        coco_info.append(mloss.item())

        print("val_loss: ", coco_info[14],"\n")
        #检查runs文件夹是否存在，若不存在则创建
        #if not os.path.exists("./runs"):
        #    os.makedirs("./runs")
        # 将实验结果写入txt，保存在runs文件夹下
        if args.with_FFCA:
            results_file = "{}/withFFCA_results.txt".format(args.log_path)  # 修改保存结果的文件路径为"./runs/results.txt"
        else:
            results_file = "{}/noFFCA_results.txt".format(args.log_path)
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # @0.5 mAP
        s1_abs_error.append(coco_info[10])
        sc_abs_error.append(coco_info[11])
        fh1_abs_error.append(coco_info[12])
        fh2_abs_error.append(coco_info[13])
        val_loss.append(coco_info[14])
        if check_loss_list(s1_abs_error, s1_abs_error[-1]):
            best_err[0] = s1_abs_error[-1]
        if check_loss_list(sc_abs_error, sc_abs_error[-1]):
            best_err[1] = sc_abs_error[-1]
        if check_loss_list(fh1_abs_error, fh1_abs_error[-1]):
            best_err[2] = fh1_abs_error[-1]
        if check_loss_list(fh2_abs_error, fh2_abs_error[-1]):
            best_err[3] = fh2_abs_error[-1]

        is_save = check_loss_list(val_loss, val_loss[-1])
        wandb_log(epoch, train_loss[-1], coco_info, best_err)
        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        if is_save:
            best_model = save_files
            best_epoches = epoch
            # torch.save(save_files, "./save_weights/model-{}.pth".format(epoch))
        if epoch == (args.epochs - 1):
            last_model = save_files

    #save best model and last model
    if best_epoches == (args.epochs - 1):
        torch.save(best_model, "{}/best_model.pth".format(args.output_dir))
    else:
        torch.save(best_model, "{}/best_model-{}.pth".format(args.output_dir ,best_epoches))
        torch.save(last_model, "{}/last_model-{}.pth".format(args.output_dir,epoch))
    #save train params
    train_params = {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_epochs": args.epochs,
        "weight-decay": args.weight_decay,
        "lr_steps": args.lr_steps,
        # 添加其他训练参数...
    }
    wandb.run.save()
    wandb.finish()
    with open("{}/train_config.txt".format(args.log_path), "w") as f:
        for key, value in train_params.items():
            f.write(f"{key}: {value}\n")

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, val_loss, learning_rate,str(args.log_path))

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
    parser.add_argument('--fixed-size', default=[256,256], nargs='+', type=int, help='input size')
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
    parser.add_argument('--lr-steps', default=[100, 150], nargs='+', type=int, help='decrease lr every step-size epochs')
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
    parser.add_argument("--log-path", default = "./runs/exp", help="log path")
    parser.add_argument("--with_FFCA", default= True , help="enable FFCA")
    parser.add_argument("--optimizer", default="adamw", help="optimizer")
    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    args.output_dir = increment_path(Path(args.output_dir),Path(args.log_path), exist_ok=False,mkdir=True)
    args.log_path = increment_path(Path(args.log_path),Path(args.outpur_dir), exist_ok=False,mkdir=True)
    #args.fixed_size = [args.fixed_size[0], args.fixed_size[0]]
    #steps = args.lr_steps[0]
    #next_steps = steps + 50
    #args.lr_steps = [steps, next_steps]
    main(args)

