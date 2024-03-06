import torch
import math
import numpy as np

class KpLoss(object):
    def __init__(self, centerloss = False):
        self.centerloss = centerloss
        self.criterion = torch.nn.MSELoss(reduction='none')
    def __call__(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0]#获取batch size大小
        # [num_kps, H, W] -> [B, num_kps, H, W]
        #遍历每张图片的Heatmap和target进行计算
        #利用stack将每张图片的Heatmap和target进行堆叠
        #print("targets:",targets
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
        # [num_kps] -> [B, num_kps]
        kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])
                # [B, num_kps, H, W] -> [B, num_kps]
        #logits为网络预测的Heatmap，heatmap为网格法计算的Heatmap
        if self.centerloss:
            loss = centerloss(logits, heatmaps,kps_weights)
        else:
            loss = self.criterion(logits, heatmaps).mean(dim=[2, 3])#在H和W维度上求均值,得到尺寸为[B*C]
        loss = torch.sum(loss) / bs
        return loss
def centerloss(logits,heatmaps,kps_weights):
    #input should be the size of B*C*H*W
        criterion = torch.nn.MSELoss(reduction='none')
        pos_inds = heatmaps.eq(1).float()
        #print("pos_inds:",pos_inds.shape)
        # pos_inds = gt.gt(0.999).float()
        neg_inds = heatmaps.lt(1).float()
        #越接近中心点，权重越小
        neg_weights = torch.exp(1-heatmaps)
        #print("neg_weights:",neg_weights.shape)

        loss = 0
            
        #logits = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1 - 1e-4)
        #print("logit:",logits.shape)
        #只关注关键点位置的损失
        #TODO:不再是成以五倍的权重，而是以关键点权重为权重，关键点周围特征越明显权重越高，越不明显权重越低
        pos_loss = criterion(logits,heatmaps) * pos_inds
        #print("pos_loss:",pos_loss.shape)
        #非关键点位置损失，辅助判断,非关键点处应全为0，越靠近中心权重越小
        #非中心处，logits越大，mseloss越大
        neg_loss = criterion(1-logits,neg_inds) * neg_weights * neg_inds
        pos_loss = pos_loss.mean(dim=[2, 3]) * 2 * kps_weights
        neg_loss = neg_loss.mean(dim=[2, 3])
        #print(len(logits))

        loss = (pos_loss + neg_loss)
        return loss

class AW_loss(object):
    def __init__(self):
                #hyeper params 
        self.omega = 14
        self.alpha = 2.5
        self.epsilon = 1
        self.theta  = 0.5
        self.gamma = 1e-2
    def linear_part(self,logits,targets):
        A = self.omega*(1/(1+(self.theta/self.epsilon)**(self.alpha-targets)))*(self.alpha-targets)*((self.theta/self.epsilon)**(self.alpha-targets-1)*(1/self.epsilon))
        C = self.theta*A-self.omega*torch.log(1+((self.theta/self.epsilon)**(self.alpha-targets)))
        return A*abs(logits-targets)-C
    
    def nonlinear_part(self,logits,targets):
        return self.omega*torch.log(1+(abs((logits-targets)/self.epsilon))**(self.alpha-targets))

    def __call__(self, logits, targets,decay = 1.0,amp = 1.0):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        #print("aw loss enabled")
        device = logits.device
        bs = logits.shape[0]#获取batch size大小
        # [num_kps, H, W] -> [B, num_kps, H, W]
        #遍历每张图片的Heatmap和target进行计算
        #利用stack将每张图片的Heatmap和target进行堆叠
        #print("targets:",targets
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
        # [num_kps] -> [B, num_kps]
        #kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])
                # [B, num_kps, H, W] -> [B, num_kps]
        #logits为网络预测的Heatmap，heatmap为网格法计算的Heatmap
        diff_map = abs(logits-heatmaps)
        loss = 0
        loss = torch.where(diff_map >= (self.theta**amp), self.linear_part(logits,heatmaps*amp), self.nonlinear_part(logits,torch.pow(heatmaps,amp)))
        loss = torch.sum(loss) / bs
        return loss
   
