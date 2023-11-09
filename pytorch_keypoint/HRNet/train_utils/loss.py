import torch


class KpLoss(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0]#获取batch size大小
        # [num_kps, H, W] -> [B, num_kps, H, W]
        #遍历每张图片的Heatmap和target进行计算
        #利用stack将每张图片的Heatmap和target进行堆叠
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
        # [num_kps] -> [B, num_kps]
        kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])

        # [B, num_kps, H, W] -> [B, num_kps]
        #logits为网络预测的Heatmap，heatmap为网格法计算的Heatmap
        loss = self.criterion(logits, heatmaps).mean(dim=[2, 3])#在H和W维度上求均值
        loss = torch.sum(loss * kps_weights) / bs
        return loss
