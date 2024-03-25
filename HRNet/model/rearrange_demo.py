from einops import rearrange
import torch


demo = torch.arange(36).view(9, 2, 2)
print('demo: ', demo)
out = rearrange(demo, '(k1 k2) w h -> 1 (w k1) (h k2)', k1=3, k2=3)
print('end=', out)
