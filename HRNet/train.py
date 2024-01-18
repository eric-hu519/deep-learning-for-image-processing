import json
import os
import datetime
from pathlib import Path
import torch
from torch.utils import data
import numpy as np
import glob
import transforms
from model import HighResolutionNet
from my_dataset_coco import CocoKeypoint
from mydataset_kfold import MyDataset
from train_utils import train_eval_utils as utils
import wandb
from train_utils import logutils

import random

def create_model(num_joints, load_pretrain_weights=False, with_FFCA=False, spatial_attention=False, skip_connection=False,swap_att=False):
    #base_channel=32 means HRnet-w32
    model = HighResolutionNet(base_channel=32, num_joints=num_joints, with_FFCA=with_FFCA, spatial_attention=spatial_attention, skip_connection=skip_connection,swap_att=swap_att)
    
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

    #missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    #if len(missing_keys) != 0:
        #print("missing_keys: ", missing_keys, "\n", "unexpected_keys: ", unexpected_keys)
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

def cross_validate(args = None):
    num_kfold = 10
    #如果args为空，则为sweep模式
    if args is None:
        sweep_run = wandb.init()
        sweep_id = sweep_run.sweep_id or "unknown"
        print("sweep_id: ",sweep_id,"\n")
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
    #DEBUG模式下,args.debug = True
    else:
        sweep_id = 'unknown'
        sweep_run_name = 'unknown_2'
        sweep_run_id = 'unknown_test'
        config = parameters_dict
        config['data-path'] = 'datasets'
    dataset = CocoKeypoint(config['data-path'], "allanno", transforms=None, fixed_size=config['fixed-size'])
    print("len of dataset: ",len(dataset),"\n")
    # Perform k-fold cross-validation
    kf = KFold(n_splits= num_kfold,shuffle=True,random_state=3407)
    metrics = []
    save_path = []
    num = 0
    failed_fold = []
    for train_index, val_index in kf.split(dataset):
        #reset env for each fold
        if args is None:
            logutils.reset_wandb_env()
        
        print("current val \n", val_index)
        # print(config['fixed-size'])
        # Create train and validation datasets based on the fold indices
        train_dataset = data.Subset(dataset, train_index)
        val_dataset = data.Subset(dataset, val_index)
        num += 1
        try:
            # Train the model using the train dataset
            result = train(
                sweep_id=sweep_id,
                num=num,
                sweep_run_name=sweep_run_name,
                config=config,  # Specify training parameters
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                args=args,
                train_index = train_index,
                val_index = val_index,
                metrics = metrics,
                save_path = save_path,
            )
            metrics.append(result[0])
            save_path.append(result[1])
           
            print("fold {} completed!".format(num),"\n", "accuray: ",result[0],"\n")
            
            
        #terminate current fold if training failed
        except logutils.TrainingException:
            print("Training failed for fold {}. Starting next fold.".format(num))
            failed_fold.append(num)
            
    if len(metrics) == 0:
        val_accuracy = -1
        print("metrics is empty! Cross Val FAILED!!!!!!!!")
        print("finishing current sweep run...")
        # Resume the sweep run
        sweep_run = wandb.init(id=sweep_run_id, resume="must")
        # Log metric to sweep run
        sweep_run.log(dict(val_accuracy =val_accuracy))
        sweep_run.finish()
        
    else:
        val_accuracy=sum(metrics) / len(metrics)
        #get the postion of the minimum number in metrics
        #将failed fold插入到metrics中，防止best_fold位置错误
        if len(failed_fold) != 0:   
            for i in failed_fold:
                metrics.insert(i-1,-1)
        #get the best fold
        best_fold = metrics.index(min(metrics))+1
        
        
        print("Cross validation COMPLETE!! val_accuracy: ",val_accuracy,"\n","bets fold is {}".format(best_fold))
        if len(failed_fold) != 0:
            print("failed fold: ",failed_fold)
        #save metrics and val_accuracy as txt
        with open("sweep_log/run-{}-metrics.txt".format(sweep_run_id), "a") as f:
            f.write("metrics: {}\n".format(metrics))
            f.write("val_accuracy: {}\n".format(val_accuracy))
            #记录失败的fold
            if len(failed_fold) != 0:
                f.write("failed_fold: {}\n".format(failed_fold))
    if args is None: 
        print("finishing current sweep run...")
        # Resume the sweep run
        sweep_run = wandb.init(id=sweep_run_id, resume="must")
        # Log metric to sweep run
        sweep_run.log(dict(val_accuracy =val_accuracy))
        sweep_run.finish()

        print("*" * 40)
        print("Sweep URL:       ", sweep_url)
        print("Sweep Group URL: ", sweep_group_url)
        print("*" * 40)





def train(num, 
          sweep_id, 
          sweep_run_name,
          config,
          train_dataset,
          val_dataset,
          args = None,
          train_index = None,
          val_index = None, 
          metrics = None, 
          save_path = None):
    #set run_name for each fold
    run_name = f'{sweep_run_name}-{num}'
    
    if args is None:
        run = wandb.init(
            group=sweep_id,
            job_type=sweep_run_name,
            name=run_name,
            config=config,
            reinit=True
        )
    else:
        #set params for debug only
        sweep_id = 'unknown_test'
        sweep_run_name = 'unknown_2'
        config = parameters_dict
        config['lr'] = 0.02
        config['wd'] = 1e-4
        config['lr-steps'] = 2
        config['fixed-size'] = 512

        config['lr-gamma'] = 0.1
        config['device'] = 'cuda:0'
        config['epochs'] = 210
        config['num_joints'] = 4
        config['data-path'] = 'datasets'
        config['keypoints_path'] = './spinopelvic_keypoints.json'
        config['output-dir'] = './save_weights/exp'
        config['amp'] = True
        config['savebest'] = True
        config['resume'] = ''
        config['with_FFCA'] = False
        config['start-epoch'] = 0
        config['s1_weight'] = 4.5
        config['sc_weight'] = 5
        config['fh1_weight'] = 2
        config['fh2_weight'] = 5
    #convert config to args
    if isinstance(config['fixed-size'],list):
        config['fixed-size'] = config['fixed-size'][0]
    run_config = logutils.sweep_override(config)
    #print("run_config: ",run_config,"\n")
    device = torch.device(run_config['device'] if torch.cuda.is_available() else "cpu")


    # save coco_info
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    print("*" * 40)
    print("*" * 40)
    with open(run_config['keypoints_path'], "r") as f:
        person_kps_info = json.load(f)
        #sweep for best weights by setting weights of keypoints
        
        person_kps_info["kps_weights"][0] = run_config['s1_weight']
        person_kps_info["kps_weights"][1] = run_config['sc_weight']
        person_kps_info["kps_weights"][2] = run_config['fh1_weight']
        person_kps_info["kps_weights"][3] = run_config['fh2_weight']
        print("keypoints weights: ",person_kps_info["kps_weights"],"\n")
    #get image size info
    fixed_size = run_config['fixed-size']
    #get heatmap size info(which is 1/4 of image size) 
    heatmap_hw = (run_config['fixed-size'][0] // 4,run_config['fixed-size'][1] // 4)
    kps_weights = np.array(person_kps_info["kps_weights"],
                           dtype=np.float32).reshape((run_config['num_joints'],))
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
    #对训练数据集和测试数据集进行预处理
    train_dataset = MyDataset(run_config['data-path'],train_dataset, transform=data_transform["train"],train_ids=train_index)
    val_dataset = MyDataset(run_config['data-path'],val_dataset, transform=data_transform["val"],train_ids=val_index)

    print("len of train_dataset: ",len(train_dataset),"\n",
              "len of val_dataset: ",len(val_dataset),"\n")
    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = run_config['batch_size']
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
    #val_dataset = CocoKeypoint(data_root, "val", transforms=data_transform["val"], fixed_size=config['fixed-size'],
                               #det_json_path=args.person_det)
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=nw,
                                      collate_fn=val_dataset.collate_fn)

    # create model
    #model = create_model(num_joints=run_config['num_joints'], with_FFCA=run_config['with_FFCA'])
    model = create_model(num_joints=run_config['num_joints'])
    # print(model)

    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    #define adam as optimizer
    optimizer = torch.optim.AdamW(params,
                                  lr=run_config['lr'],
                                  weight_decay=run_config['wd'])

    scaler = torch.cuda.amp.GradScaler() if run_config['amp'] else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=run_config['lr-steps'], gamma=config['lr-gamma'])

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if run_config['resume'] != "":
        checkpoint = torch.load(run_config['resume'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        run_config['start-epoch'] = checkpoint['epochs'] + 1
        if run_config['amp'] and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(run_config['start-epoch']))

    train_loss = []
    learning_rate = []
    val_map = []
    sc_abs_error = []
    s1_abs_error = []
    fh1_abs_error = []
    fh2_abs_error = []
    sc_std = []
    s1_std = []
    fh1_std = []
    fh2_std = []
    val_loss = []
    best_err = np.zeros((4,))
    is_last_epoch = False
    

#训练主函数
    for epoch in range(run_config['start-epoch'], run_config['epochs']):
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
        if epoch == (run_config['epochs'] - 1):
            is_last_epoch = True
        # evaluate on the test dataset
        coco_info = utils.evaluate(model, val_data_loader, device=device,
                                   flip=True, is_last_epoch=is_last_epoch, save_dir=str(run_config['last-dir']))
        coco_info.append(mloss.item())

        print("val_loss: ", coco_info[-1],"\n")
        #检查runs文件夹是否存在，若不存在则创建
        #if not os.path.exists("./runs"):
        #    os.makedirs("./runs")
        # 将实验结果写入txt，保存在runs文件夹下
        if run_config['with_FFCA']:
            results_file = "{}/withFFCA_results_fold{}.txt".format(run_config['last-dir'], num)  # 修改保存结果的文件路径为"./runs/results.txt"
        else:
            results_file = "{}/noFFCA_results_fold{}.txt".format(run_config['last-dir'], num)
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
        s1_std.append(coco_info[14])
        sc_std.append(coco_info[15])
        fh1_std.append(coco_info[16])
        fh2_std.append(coco_info[17])

        val_loss.append(coco_info[-1])
        if check_loss_list(s1_abs_error, s1_abs_error[-1]):
            best_err[0] = s1_abs_error[-1]
        if check_loss_list(sc_abs_error, sc_abs_error[-1]):
            best_err[1] = sc_abs_error[-1]
        if check_loss_list(fh1_abs_error, fh1_abs_error[-1]):
            best_err[2] = fh1_abs_error[-1]
        if check_loss_list(fh2_abs_error, fh2_abs_error[-1]):
            best_err[3] = fh2_abs_error[-1]
        
        is_save = check_loss_list(val_loss, val_loss[-1])
        if args is None:
            run.log(logutils.wandb_log(train_loss[-1], coco_info, best_err))
        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epochs': epoch}
        if run_config['amp']:
            save_files["scaler"] = scaler.state_dict()
        if is_save:
            best_model = save_files
            best_epoches = epoch
            # torch.save(save_files, "./save_weights/model-{}.pth".format(epoch))
        if epoch == (run_config['epochs'] - 1):
            last_model = save_files

    #val acc equals the mean error of all points of the last epoch
    val_accuracy = (s1_abs_error[-1]+sc_abs_error[-1]+fh1_abs_error[-1]+fh2_abs_error[-1])/4

    print("fold {}----val_accuracy: ".format(num),val_accuracy,"\n")
    #save best model and last model
    if run_config['savebest']:
        if num == 1:
            #save model for the first run
            if best_epoches == (run_config['epochs'] - 1):
                torch.save(best_model, "{}/best_model_fold{}.pth".format(run_config['last-dir'],num))
            else:
                torch.save(best_model, "{}/best_model-{}-fold{}.pth".format(run_config['last-dir'] ,best_epoches,num))
                torch.save(last_model, "{}/last_model-{}-fold{}.pth".format(run_config['last-dir'],epoch,num))
        elif num != 1 & (len(metrics) != 0):
            if val_accuracy <= min(metrics):
                print("new best model for fold {} saved \n".format(num))
                #save model of current fold
                if best_epoches == (run_config['epochs'] - 1):
                    torch.save(best_model, "{}/best_model_fold{}.pth".format(run_config['last-dir'],num))
                else:
                    torch.save(best_model, "{}/best_model-{}-fold{}.pth".format(run_config['last-dir'] ,best_epoches,num))
                    torch.save(last_model, "{}/last_model-{}-fold{}.pth".format(run_config['last-dir'],epoch,num))
                #remove the last model
                assert len(save_path) != 0, "save_path is empty"
                pthfile = glob.glob(os.path.join(save_path[-1],'*.pth'))
                for f in pthfile:
                    os.remove(f)
                print("new best model saved \n","removing the last best model}!")
    #if not save best then save the model for every fold
    else:
        if best_epoches == (run_config['epochs'] - 1):
            torch.save(best_model, "{}/best_model_fold{}.pth".format(run_config['last-dir'],num))
        else:
            torch.save(best_model, "{}/best_model-{}-fold{}.pth".format(run_config['last-dir'] ,best_epoches,num))
            torch.save(last_model, "{}/last_model-{}-fold{}.pth".format(run_config['last-dir'],epoch,num))
    #save train params
    train_params = {
        "fixed-size": run_config['fixed-size'],
        "batch_size": run_config['batch_size'],
        "learning_rate": run_config['lr'],
        "num_epochs": run_config['epochs'],
        "weight-decay": run_config['wd'],
        "lr_steps": run_config['lr-steps'],
        "lr_gamma": run_config['lr-gamma'],
        "keypoints weight": person_kps_info["kps_weights"]
        # 添加其他训练参数...
    }
    with open("{}/train_config_fold{}.txt".format(run_config['last-dir'],num), "w") as f:
        for key, value in train_params.items():
            f.write(f"{key}: {value}\n")

    
    if len(best_err) != 0:
        print("min_sc_e: ",best_err[0],"\n",
              "min_s1_e: ",best_err[1],"\n",
              "min_fh1_e: ",best_err[2],"\n",
              "min_fh2_e: ",best_err[3],"\n")
    print("*" * 40)
    print("*" * 40)
    #save abs_err as txt
    if len(sc_abs_error) != 0:
        with open("{}/abs_error_fold{}.txt".format(run_config['last-dir'],num), "w") as f:
            f.write("sc_abs_error: {}\n".format(sc_abs_error))
            f.write("s1_abs_error: {}\n".format(s1_abs_error))
            f.write("fh1_abs_error: {}\n".format(fh1_abs_error))
            f.write("fh2_abs_error: {}\n".format(fh2_abs_error))
            f.write("sc_std: {}\n".format(sc_std))
            f.write("s1_std: {}\n".format(s1_std))
            f.write("fh1_std: {}\n".format(fh1_std))
            f.write("fh2_std: {}\n".format(fh2_std))
    if args is None:
        run.log(dict(val_accuracy=val_accuracy))
        run.finish()
    return val_accuracy, run_config['last-dir']


#sweep configuration for wandb swe
sweep_config = {
    'method': 'bayes',
    'name': 'spine-sweep',
    'metric': {
        'goal': 'minimize',
        'name': 'val_accuracy'
    }
    
}
#set parameters for train
parameters_dict = {
    'device': {
        'values': ['cuda:0']
        },
    'data-path': {
        'values': ['datasets']
        },
    'keypoints_path': {
        'values': ['./spinopelvic_keypoints.json']
        },
    'fixed-size': {
        'values': [128,256,512,640]
        },
    'num_joints': {
        'values': [4]
        },
    'output-dir': {
        'values': ['./save_weights/exp']
        },
    'start-epoch': {
        'values': [0]
        },
    'epochs': {
        'values': [185]
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
        'values': [True]
        },
    'resume':{
        'values': ['']
        },
    'with_FFCA':{
        'values': [False]
        },
    's1_weight':{
        'values': [1,1.5,2,2.5,3,3.5,4,4.5,5]
        },  
    'sc_weight':{
        'values': [1,1.5,2,2.5,3,3.5,4,4.5,5]
        },
    'fh1_weight':{
        'values': [1,1.5,2,2.5,3,3.5,4,4.5,5]
        },
    'fh2_weight':{
        'values': [1,1.5,2,2.5,3,3.5,4,4.5,5]
        }
    }


#main function for sweep
def main(args):
    #add params to sweep_config
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sweep_config['parameters'] = parameters_dict
    if not args.debug:
        wandb.login()
        if args.resume:
            #resume mode
            #need to set resume for sweep control in wandb.ai
            sweep_id = args.sweep_id
            wandb.agent(sweep_id, function = cross_validate,project = args.project)
        else:
            sweep_id = wandb.sweep(sweep_config, project = args.project)   
            #print("sweep_id: ",sweep_id,"\n")
            #制定运行方程为cross_validate
            wandb.agent(sweep_id, function=cross_validate)
        wandb.finish()
    else:
        #只有debug模式下args才不为空
        cross_validate(args)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--debug', default=True, help='debug mode')
    parser.add_argument('--resume', default=False, help='resume mode')
    parser.add_argument('--sweep_id', default='f5cyaw6i', help='sweep id')
    parser.add_argument('--project',default='Spine-final',help='project name')
    args = parser.parse_args()
    main(args)


    
