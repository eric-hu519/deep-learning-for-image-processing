import torch
from model import HighResolutionNet
model = HighResolutionNet(base_channel=32, num_joints=4, with_FFCA=True)
        # 链接:https://pan.baidu.com/s/1Lu6mMAWfm_8GGykttFMpVw 提取码:f43o
weights_dict = torch.load("./save_weights/exp55/best_model-175.pth", map_location='cpu')
missing_keys, unexpected_keys = model.load_state_dict(weights_dict['model'], strict=False)
for k in list(weights_dict['model']):
    print(k)
    