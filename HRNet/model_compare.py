


def main(args):
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    logdir = args.logdir
    ids = args.ids.split(' ')
    metrics = args.metrics.split(' ')
    angles = ['ss_angle', 'pt_angle', 'pi_angle']
    keypoints = ['sc', 's1', 'fh1', 'fh2']
    metrics_dict = {}
    config_dict_list = []
    now = datetime.datetime.now()
    save_dir = args.save_dir + '/' + now.strftime("%m%d%H%M")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for id in ids:
        test_logdir = logdir+id
        if not os.path.exists(test_logdir):
            raise ValueError(f'logdir {test_logdir} does not exist')
        if not os.path.exists(test_logdir+'/train_config_fold0.txt'):
            raise ValueError(f'logdir {test_logdir} does not have training config file')
        #read config file
        config_file = test_logdir+'/train_config_fold0.txt'
        config_dict = read_config_to_dict(config_file)
        config_dict_list.append(config_dict)
        for metric in metrics:
            if metric not in ['angle_error', 'error', 'CMAE', 'SMAE', 'ED']:
                raise ValueError(f'metric {metric} is not supported')
            if not os.path.exists(test_logdir+'/'+metric+'.txt'):
                raise ValueError(f'logdir {test_logdir} does not have {metric}.txt')
            else:
                print(f'test{id}/{metric}.txt found')
                data_with_nan = np.loadtxt(test_logdir+'/'+metric+'.txt')
                isnan = np.isnan(data_with_nan)
                if data_with_nan.ndim > 1:
                    row_has_nan = np.any(isnan, axis=1)
                    data = data_with_nan[~row_has_nan]
                else:
                    data = data_with_nan[~isnan]
                # data = np.loadtxt(test_logdir+'/'+metric+'.txt')
                # print(data.shape)
                if data.ndim == 2:
                    if data.shape[1] == 3:
                        #get angle error
                        for i in range(data.shape[1]):
                            metrics_dict[id+'_'+angles[i]] = data[:,i]
                    elif data.shape[1] == 4:
                        #get error
                        for i in range(data.shape[1]):
                            metrics_dict[id+'_'+keypoints[i]] = data[:,i]
                else:
                    metrics_dict[id+'_'+metric] = data
    #print keys
    print(metrics_dict.keys())
    #plot angles with boxplot
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    for angle in angles:
        ax[angles.index(angle)].boxplot([metrics_dict[id+'_'+angle] for id in ids], labels=ids,showfliers=False,showmeans=True,showbox=True,meanline=True,medianprops=dict(color='r'))
        ax[angles.index(angle)].set_title(angle)
    fig.suptitle('Angle Error Boxplot')
    plt.tight_layout()
    plt.savefig(save_dir+'/angle_boxplot.png')
    if args.display:
        plt.show()
    #plot errors with boxplot
    fig, ax = plt.subplots(1,4,figsize=(20,5))
    for keypoint in keypoints:
        ax[keypoints.index(keypoint)].boxplot([metrics_dict[id+'_'+keypoint] for id in ids], labels=ids,showfliers=False,showmeans=True,showbox=True,meanline=True,medianprops=dict(color='r'))
        ax[keypoints.index(keypoint)].set_title(keypoint)
    fig.suptitle('Error Boxplot')
    plt.tight_layout()
    plt.savefig(save_dir+'/error_boxplot.png')
    if args.display:
        plt.show()
    fig, ax = plt.subplots(1,3,figsize=(20,5))
    datacom=['CMAE', 'SMAE', 'ED']
    for keypoint in datacom:
        ax[datacom.index(keypoint)].boxplot([metrics_dict[id+'_'+keypoint] for id in ids], labels=ids,showfliers=False,showmeans=True,showbox=True,meanline=True,medianprops=dict(color='r'))
        ax[datacom.index(keypoint)].set_title(keypoint)
    fig.suptitle('Evaluation and comparison')
    plt.tight_layout()
    plt.savefig(save_dir+'/Evaluatio_Comparison.png')
    if args.display:
        plt.show()

    

def read_config_to_dict(filepath):
    config_dict = {}
    with open(filepath, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            config_dict[key] = value
    return config_dict

def get_diff_config(config_dict_list):
    diff_config = set()
    for i in range(len(config_dict_list)-1):
            for key in config_dict_list[i]:
                if config_dict_list[i][key] != config_dict_list[i+1][key]:
                    diff_config.add(key)
    return diff_config



if __name__ == '__main__':
    import argparse
    import datetime
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--logdir', default='./save_weights/', help='log directory')
    parser.add_argument('--ids', default='OURS FFCA HRNet',help='model ids to compare')
    parser.add_argument('--save_dir', default='./model_compare_results',help='metrics to compare')
    parser.add_argument('--metrics', default='angle_error error CMAE SMAE ED',help='metrics to compare')
    parser.add_argument('--display',default= False, help='display the results')
    args = parser.parse_args()
    main(args)