import json
import os
import datetime
from pathlib import Path
import torch
from torch.utils import data
import numpy as np

import transforms
from model import HighResolutionNet
from my_dataset_coco import CocoKeypoint
from mydataset_kfold import myKeypoint
from train_utils import train_eval_utils as utils
import wandb
import mypredict
from train_utils import logutils
import random

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
                    #del weights_dict[k]import argparse

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


from sklearn.model_selection import KFold

def cross_validate():
    num_kfold = 10
    sweep_run = wandb.init()
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = f'{project_url}/groups/{sweep_id}'
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown_2"
    sweep_run_id = sweep_run.id
    sweep_run.finish()
    wandb.sdk.wandb_setup._setup(_reset=True)
    config = dict(sweep_run.config)

    # Read the entire dataset
    dataset = CocoKeypoint(config['data-path'], "allanno", transforms=None, fixed_size=config['fixed_size'])

    # Perform k-fold cross-validation
    kf = KFold(n_splits= num_kfold)
    metrics = []
    num = 0
    for train_index, val_index in kf.split(dataset):
        logutils.reset_wandb_env()
        
        # Create train and validation datasets based on the fold indices
        train_dataset = data.Subset(dataset, train_index)
        val_dataset = data.Subset(dataset, val_index)
        
        # Train the model using the train dataset
        result = train(
            sweep_id=sweep_id,
            num=num,
            sweep_run_name=sweep_run_name,
            config=config,  # Specify training parameters
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )
        metrics.append(result)
        num += 1

    # Resume the sweep run
    sweep_run = wandb.init(id=sweep_run_id, resume="must")
    # Log metric to sweep run
    sweep_run.log(dict(val_accuracy=sum(metrics) / len(metrics)))
    sweep_run.finish()

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)





def train(num, sweep_id, sweep_run_name,config,train_dataset,val_dataset):
    run_name = f'{sweep_run_name}-{num}'
    run = wandb.init(
        group=sweep_id,
        job_type=sweep_run_name,
        name=run_name,
        config=config,
        reinit=True
    )
    #convert config to args
    config = logutils.sweep_override(config)
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

#set random seed
    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)
    np.random.seed(3407)

    # save coco_info
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    with open(config['keypoints_path'], "r") as f:
        person_kps_info = json.load(f)
    #get image size info
    fixed_size = config['fixed_size']
    #get heatmap size info(which is 1/4 of image size) 
    heatmap_hw = (config['fixed_size'][0] // 4, config['fixed_size'][1] // 4)
    kps_weights = np.array(person_kps_info["kps_weights"],
                           dtype=np.float32).reshape((config['num_joints'],))
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
#TODO: perform data transform on train_dataset and val_dataset
    #transform datasets 
    train_dataset = train_dataset.transform(data_transform["train"])
    val_dataset = val_dataset.transform(data_transform["val"])
    #transform datasets after split

    
    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = config['']
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
    #val_dataset = CocoKeypoint(data_root, "val", transforms=data_transform["val"], fixed_size=config['fixed_size'],
                               #det_json_path=args.person_det)
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=nw,
                                      collate_fn=val_dataset.collate_fn)

    # create model
    model = create_model(num_joints=config['num_joints'], with_FFCA=config['with_FFCA'])
    # print(model)

    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    #define adam as optimizer
    optimizer = torch.optim.AdamW(params,
                                  lr=config['lr'],
                                  weight_decay=config['weight_decay'])

    scaler = torch.cuda.amp.GradScaler() if config['amp'] else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['lr-steps'], gamma=config['lr-gamma'])

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if config['resume'] != "":
        checkpoint = torch.load(config['resume'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config['start-epoch'] = checkpoint['epoch'] + 1
        if config['amp'] and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(config['start-epoch']))

    train_loss = []
    learning_rate = []
    val_map = []
    sc_abs_error = []
    s1_abs_error = []
    fh1_abs_error = []
    fh2_abs_error = []
    val_loss = []
    best_err = np.zeros((4,))

    


    for epoch in range(config['start-epoch'], config['epochs']):
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
        if config:
            results_file = "{}/withFFCA_results.txt".format(config['output-dir'])  # 修改保存结果的文件路径为"./runs/results.txt"
        else:
            results_file = "{}/noFFCA_results.txt".format(config['output-dir'])
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
        if not config['debug']:
            logutils.wandb_log(epoch, train_loss[-1], coco_info, best_err)
        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if config['amp']:
            save_files["scaler"] = scaler.state_dict()
        if is_save:
            best_model = save_files
            best_epoches = epoch
            # torch.save(save_files, "./save_weights/model-{}.pth".format(epoch))
        if epoch == (config[''] - 1):
            last_model = save_files

    #save best model and last model
    if best_epoches == (config[''] - 1):
        torch.save(best_model, "{}/best_model.pth".format(config['output-dir']))
    else:
        torch.save(best_model, "{}/best_model-{}.pth".format(config['output-dir'] ,best_epoches))
        torch.save(last_model, "{}/last_model-{}.pth".format(config['output-dir'],epoch))
    #save train params
    train_params = {
        "batch_size": config['batch_size'],
        "learning_rate": config['lr'],
        "num_epochs": config['epochs'],
        "weight-decay": config['weight_decay'],
        "lr_steps": config['lr_steps'],
        # 添加其他训练参数...
    }
    if not config['debug']:
        wandb.run.save()
        wandb.finish()
    with open("{}/train_config.txt".format(config['output-dir']), "w") as f:
        for key, value in train_params.items():
            f.write(f"{key}: {value}\n")

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, val_loss, learning_rate,str(config['output-dir']))

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map,str(config['output-dir']))
    #plot abs error curve
    if len(sc_abs_error) != 0:
        from plot_curve import plot_abs_error
        plot_abs_error(sc_abs_error,'sc',str(config['output-dir']))
        plot_abs_error(s1_abs_error,'s1',str(config['output-dir']))
        plot_abs_error(fh1_abs_error,'fh1',str(config['output-dir']))
        plot_abs_error(fh2_abs_error,'fh2',str(config['output-dir']))
    if len(best_err) != 0:
        print("min_sc_e: ",best_err[0],"\n",
              "min_s1_e: ",best_err[1],"\n",
              "min_fh1_e: ",best_err[2],"\n",
              "min_fh2_e: ",best_err[3],"\n")
    val_accuracy = random.random()
    run.log(dict(val_accuracy=val_accuracy))
    run.finish()
    return val_accuracy


#sweep configuration for wandb swe
sweep_config = {
    'method': 'grid',
    'name': 'sweep-test-1',
    'metric': {
        'goal': 'maximize',
        'name': 'val_accuracy'
    }
    
}
#set parameters for train
parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
        },
    'fc_layer_size': {
        'values': [128, 256, 512]
        },
    'dropout': {
          'values': [0.3, 0.4, 0.5]
        },
    'device': {
        'values': ['cuda:0']
        },
    'data-path': {
        'values': ['datasets']
        },
    'keypoints-path': {
        'values': ['./spinopelvic_keypoints.json']
        },
    'fixed-size': {
        'values': [128, 256, 512, 640]
        },
    'num-joints': {
        'values': [4]
        },
    'output-dir': {
        'values': ['./save_weights/exp']
        },
    'start-epoch': {
        'values': [0]
        },
    'epochs': {
        'values': [200]
        },
    'lr-steps': {
        'values': [1, 2, 3, 4, 5, 6]
        },
    'lr-gamma': {
        'distribution': 'uniform',
        'min': 0.1,
        'max': 0.8
        },
    'lr': {
        'distribution': 'uniform',
        'min': 0.00085,
        'max': 0.01
        },
    'wd': {
        'distribution': 'uniform',
        'min': 0.0005,
        'max': 0.005
        },
    'amp': {
        'values': [True]
        },
    'savebest': {
        'values': [True]
        },
    'debug':{
        'values': [False]
        },
    'resume':{
        'values': ['']
        },
    'with_FFCA':{
        'values': [True]
        }
    }
#add params to sweep_config
sweep_config['parameters'] = parameters_dict
#main function for sweep
def main():
    
    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project='sweep-test')
    #制定运行方程为cross_validate
    wandb.agent(sweep_id, function=cross_validate)
    wandb.finish()
if __name__ == '__main__':
    main()


    
