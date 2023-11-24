import datetime
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_loss_and_lr(train_loss, learning_rate, path):
    try:
        x = list(range(len(train_loss)))
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(x, train_loss, 'r', label='loss')
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        ax1.set_title("Train Loss and lr")
        plt.legend(loc='best')

        ax2 = ax1.twinx()
        ax2.plot(x, learning_rate, label='lr')
        ax2.set_ylabel("learning rate")
        ax2.set_xlim(0, len(train_loss))  # 设置横坐标整数间隔
        plt.legend(loc='best')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

        fig.subplots_adjust(right=0.8)  # 防止出现保存图片显示不全的情况
        #判断runs文件夹是否存在，不存在则创建
        path = path + '/figs/'
        if not os.path.exists(path):
            os.mkdir(path)
        fig.savefig(path+'loss_and_lr{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        plt.close()
        print("successful save loss curve! ")
    except Exception as e:
        print(e)


def plot_map(mAP,path):
    try:
        x = list(range(len(mAP)))
        plt.plot(x, mAP, label='mAp')
        plt.xlabel('epoch')
        plt.ylabel('mAP')
        plt.title('Eval mAP')
        plt.xlim(0, len(mAP))
        plt.legend(loc='best')
        path = path + '/figs/'
                #判断runs文件夹是否存在，不存在则创建
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(path + 'mAP.png')
        plt.close()
        print("successful save mAP curve!")
    except Exception as e:
        print(e)

def plot_abs_error(abs_error, err_name, path):
    try:
        x = list(range(len(abs_error)))
        plt.plot(x, (np.log(abs_error) + np.spacing(1)), label='abs_error')
        plt.xlabel('epoch')
        plt.ylabel('abs_error')
        plt.title('Eval abs_error')
        plt.xlim(0, len(abs_error))
        plt.legend(loc='best')
                #判断runs文件夹是否存在，不存在则创建
        path = path + '/figs/'
        if not os.path.exists(path):
            os.mkdir(path)
        
        plt.savefig('{}{}_abs_log_error.png'.format(path,err_name))
        
        plt.savefig('{}{}_abs_error.png'.format(path,err_name))
        plt.close()
        print("successful save {}_error curve!".format(err_name))
    except Exception as e:
        print(e)