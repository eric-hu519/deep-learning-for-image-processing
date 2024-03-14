import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
BN_MOMENTUM = 0.1

from .model_utils import Bottleneck, StageModule, myDecoder
#num_joints,表示关节点个数
class HighResolutionNet(nn.Module):
    def __init__(self, base_channel: int = 32, num_joints: int = 4, 
                 with_FFCA = True, skip_connection = True, 
                 use_rfca = True, mix_c = True,
                 pag_fusion = True, my_fusion = True):
        super().__init__()
        # Stem
        self.with_FFCA = with_FFCA
        self.skip_connection = skip_connection
        self.mix_c = mix_c
        self.use_rfca = use_rfca
        self.pag_fusion = pag_fusion
        self.my_fusion = my_fusion

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

        # Stage1
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        )
        #layer1用来调整通道数
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )
        #第一个分支上的conv不会改变特征图高宽
        #第二个分支上的conv会将特征图高宽减半
        #每新增一个分支，通道个数都会是上一个分支的两倍
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                #RFCAConv(256, base_channel,3,1),
                #ScConv(256),
                nn.Conv2d(256, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(
                    #ScConv(256),
                        # 这里又使用一次Sequential是为了适配原项目中提供的权重
                    #RFCAConv(256, base_channel * 2,3,2),
                    nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage2
        self.stage2 = nn.Sequential(
            StageModule(input_branches=2, output_branches=2, c=base_channel,use_rfca = self.use_rfca, mix_c=self.mix_c),
        )
        # transition2
        self.transition2 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    #ScConv(base_channel * 2),
                    #RFCAConv(base_channel * 2, base_channel * 4,3,2),
                    nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage3
        self.stage3 = nn.Sequential(
            StageModule(input_branches=3, output_branches=3, c=base_channel,use_rfca = self.use_rfca, mix_c=self.mix_c),
            StageModule(input_branches=3, output_branches=3, c=base_channel,use_rfca = self.use_rfca, mix_c=self.mix_c),
            StageModule(input_branches=3, output_branches=3, c=base_channel,use_rfca = self.use_rfca, mix_c=self.mix_c),
            StageModule(input_branches=3, output_branches=3, c=base_channel,use_rfca=self.use_rfca,  mix_c=self.mix_c)
        )

        self.transition3 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    #ScConv(base_channel * 4),
                    #RFCAConv(base_channel * 4, base_channel * 8,3,2),
                    nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ]) 

        # Stage4
        # 注意，最后一个StageModule只输出分辨率最高的特征层
        self.stage4 = nn.Sequential(
            StageModule(input_branches=4, output_branches=4, c=base_channel,use_rfca=self.use_rfca,  mix_c=self.mix_c),
            StageModule(input_branches=4, output_branches=4, c=base_channel,use_rfca=self.use_rfca,  mix_c=self.mix_c),
            
            #将四个输入分支进行融合，输出一个分支
            #StageModule(input_branches=4, output_branches=1, c=base_channel)
        )
        if not self.with_FFCA:
            self.stage4.add_module("StageModule",StageModule(input_branches=4, output_branches=1, c=base_channel,use_rfca=False, mix_c=self.mix_c))
        else:
            self.decoder = myDecoder(base_channel, self.use_rfca,self.mix_c,self.pag_fusion,self.my_fusion)
        
        # Final layer，通道个数要与num_joints一致
        self.final_layer = nn.Conv2d(base_channel, num_joints, kernel_size=1, stride=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)

         # 用于后续的skip connection
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list
        #print("x.shape: ", x[0].shape)
        skip_1 = x[0]
        skip_2 = x[1]  # 用于后续的skip connection
        x = self.stage2(x)
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only
        skip_3 = x[2]  # 用于后续的skip connection

        x = self.stage3(x)
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1]),
        ]  # New branch derives from the "upper" branch only

        skip_4 = x[3]  # 用于后续的skip connection

        x = self.stage4(x)
        #print("x[0]: ", x[0].shape, "\n", "x[1]: ", x[1].shape, "\n", "x[2]: ", x[2].shape, "\n", "x[3]: ", x[3].shape, "\n")

        if self.skip_connection:
            x[0] = skip_1 + x[0]
            if self.with_FFCA:
                x[1] = skip_2 + x[1]
                x[2] = skip_3 + x[2]
                x[3] = skip_4 + x[3]
        if self.with_FFCA:
            x = self.decoder(x)
            x = self.final_layer(x)
        else:
        #由于最后一层只输出一个分支，所以这里只取最后一个分支x[0]
            x = self.final_layer(x[0])
        #x = self.sigmoid(x)

        return x