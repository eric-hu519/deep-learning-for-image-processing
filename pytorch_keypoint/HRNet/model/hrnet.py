import torch.nn as nn

BN_MOMENTUM = 0.1

# 用于构建stage的block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
#这里的forward函数是用来构建block的
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class StageModule(nn.Module):
    def __init__(self, input_branches, output_branches, c):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            w = c * (2 ** i)  # 对应第i个分支的通道数
            branch = nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w)
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                else:  # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                #进行下采样，经过这个循环构建出红色的CONV结构
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    # 最后一个卷积层不仅要调整通道，还要进行下采样，构建橙色的卷积部分
                    ops.append(
                        nn.Sequential(
                            #构建下采样，调整通道个数
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # 接着融合不同尺寸信息
        x_fused = []
        #i是输出分支的索引，j是输入分支的索引
        #i=j,不做任何处理，i<j,对输入分支j进行通道调整以及上采样，方便后续相加;
        # i>j,对输入分支j进行通道调整以及下采样，方便后续相加
        for i in range(len(self.fuse_layers)):
            x_fused.append(
                self.relu(
                    #对每个分支进行融合
                    sum([self.fuse_layers[i][j](x[j]) for j in range(len(self.branches))])
                )
            )

        return x_fused

#num_joints,表示关节点个数
class HighResolutionNet(nn.Module):
    def __init__(self, base_channel: int = 32, num_joints: int = 4):
        super().__init__()
        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

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
                nn.Conv2d(256, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(  # 这里又使用一次Sequential是为了适配原项目中提供的权重
                    nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage2
        self.stage2 = nn.Sequential(
            StageModule(input_branches=2, output_branches=2, c=base_channel)
        )

        # transition2
        self.transition2 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage3
        self.stage3 = nn.Sequential(
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel)
        )

        # transition3
        self.transition3 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage4
        # 注意，最后一个StageModule只输出分辨率最高的特征层
        self.stage4 = nn.Sequential(
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            #将四个输入分支进行融合，输出一个分支
            #StageModule(input_branches=4, output_branches=1, c=base_channel)
        )
        self.decoder = myDecoder()
        
        # Final layer，通道个数要与num_joints一致
        self.final_layer = nn.Conv2d(base_channel, num_joints, kernel_size=1, stride=1)

    def forward(self, x):
        print("x.shape: ", x.shape,"\n")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list

        x = self.stage2(x)
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1]),
        ]  # New branch derives from the "upper" branch only

        x = self.stage4(x)
        print("x[0]: ", x[0].shape, "\n", "x[1]: ", x[1].shape, "\n", "x[2]: ", x[2].shape, "\n", "x[3]: ", x[3].shape, "\n")
        x = self.decoder(x)
        #由于最后一层只输出一个分支，所以这里只取最后一个分支x[0]
        x = self.final_layer(x[0])
        x = self.sigmoid(x)

        return x

import torch
import torch.nn.functional as F

class FFCA(nn.Module):
    """
    FFCA - Fusion Feature Channel Attention Module
    """
    def __init__(self, channel, reduction=16):
        super(FFCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP (implemented using Conv1d for parameter sharing)
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x should be in the shape of (batch_size, channel, height, width)
        batch, channel, _, _ = x.size()

        # Average and Max Pooling
        avg_pool = self.avg_pool(x).view(batch, channel, 1)  # shape: (batch, channel, 1)
        max_pool = self.max_pool(x).view(batch, channel, 1)  # shape: (batch, channel, 1)

        # Shared MLP for both pooling results
        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)

        # Element-wise sum and activation
        out = self.sigmoid(avg_out + max_out).view(batch, channel, 1, 1)

        return out * x

class Decoder(nn.Module):
    """
    Decoder with FFCA modules.
    """
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.ffca1 = FFCA(channels[3])
        self.ffca2 = FFCA(channels[2])
        self.ffca3 = FFCA(channels[1])
        self.ffca4 = FFCA(channels[0])
        
        # Decoder layers
        self.up1 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.up3 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.conv1 = nn.Conv2d(channels[2], channels[2], 3, padding=1)
        self.conv2 = nn.Conv2d(channels[1], channels[1], 3, padding=1)
        self.conv3 = nn.Conv2d(channels[0], channels[0], 3, padding=1)



        # Final layer to output the center point heatmap and vector map
        self.final_center = nn.Conv2d(channels[0], 1, 1)
        #self.final_vector = nn.Conv2d(channels[3], 2, 1)

    def forward(self, x):
        # Assuming x is the output of the encoder with skip connections
        #encoder_output, skip1, skip2, skip3 = x

        branch3, branch2, branch1, branch0 = x #将四个分支的输出分别赋值给branch3, branch2, branch1, branch0，分别为高分辨率到低分辨率
        

        up_branch0 = self.upsample(branch0) #up_branch0 的 H与W应该与branch1的H与W相同
        up_branch1 = self.upsample(branch1)
        up_branch2 = self.upsample(branch2)
        
        # Apply FFCA modules and upsample
        up1 = self.up1(F.relu(self.ffca1(branch0)))
        up2 = self.up2(F.relu(self.ffca2(self.conv1(up1 + branch1))))
        up3 = self.up3(F.relu(self.ffca3(self.conv2(up2 + branch2))))
        decoder_output = F.relu(self.ffca4(self.conv3(up3 + branch3)))

        # Output layers
        center_map = self.final_center(decoder_output)
        #vector_map = self.final_vector(decoder_output)

        return center_map


class myFFCA(nn.Module):
    """
    FFCA - Fusion Feature Channel Attention Module
    """
    def __init__(self, low_branch,high_branch):#输入为低级分支的通道个数
        super().__init__()
        #low_branch size should be (batch_size, 2*channel, height/2, width/2)
        
        self.low_branch = low_branch
        self.high_branch = high_branch
        self.low_channle = low_branch.size(1)
        self.high_channle = high_branch.size(1)
        #process input branch
        self.inbranch_process = nn.Sequential(
            #upsample, make lowchannel's H,W double and equals to highchannel's H,W
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            #3*3conv, make lowchannel's channel equals to highchannel's channel
            nn.Conv2d(self.low_channle, self.high_channle,3,padding=1)
        )
        # Average and Max Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#tensor size should be (batch_size, channel, 1, 1) after pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Linear(self.low_channle, self.low_channle//2)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Sequential(
            #3*3conv
            nn.Conv2d(self.low_channle, self.high_channle,3,padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self):
        up_lowbranch = self.inbranch_process(self.low_branch)
        #concatenate two branch
        concat_branch = torch.cat((up_lowbranch, self.high_branch), 1)#channle*2
        maxpooled_branch = self.max_pool(concat_branch)
        avgpooled_branch = self.avg_pool(concat_branch)

        #change size from[batch_size, channel, 1, 1] to [batch_size, channel]
        maxpooled_branch_flat = maxpooled_branch.view(maxpooled_branch.size(0), -1)
        avgpooled_branch_flat = avgpooled_branch.view(avgpooled_branch.size(0), -1)

        # Pass the tensors through the fully connected layer
        feature_weight = self.sigmoid(self.fc(maxpooled_branch_flat) + self.fc(avgpooled_branch_flat))
        #restore the size of weight from[batch_size, channel] to [batch_size, channel, 1, 1]
        feature_weight = feature_weight.view(feature_weight.size(0), feature_weight.size(1), 1, 1)

        weighted_branch = feature_weight * concat_branch

        output_branch = torch.cat((weighted_branch, self.high_branch), 1)#channle*2

        output_branch = self.output(output_branch)#half the channle

        return output_branch #output size should be (batch_size, high_channel, height, width)


class myDecoder(nn.Module):
    """
    Decoder with FFCA modules.
    """
    def __init__(self,low_branch,high_branch):
        super().__init__()
        #from lower ffca to higher ffca
        self.ffca1 = myFFCA(low_branch,high_branch)
        self.ffca2 = myFFCA(low_branch,high_branch)
        self.ffca3 = myFFCA(low_branch,high_branch)
        
    def forward(self, x):
        branch3, branch2, branch1, branch0 = x
        ffca_output1 = self.ffca1(branch0, branch1)
        ffca_output2 = self.ffca2(ffca_output1, branch2)
        ffca_output3 = self.ffca3(ffca_output2, branch3)

        return ffca_output3


