import torch.nn as nn
import torch

from einops import rearrange
BN_MOMENTUM = 0.1

# 用于构建stage的block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_rfca = True, all_rfca = True, mix_c =True):
        super(BasicBlock, self).__init__()
        if use_rfca & all_rfca:
            self.conv1 = RFCAConv(inplanes, planes,3,stride,mix_c=mix_c)
        else:
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
    def __init__(self, input_branches, output_branches, c, use_rfca = True, all_rfca = True, mix_c = True):
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
            #only 2 basic block when use rfca
            if use_rfca:
                branch  = nn.Sequential(
                    BasicBlock(w, w, use_rfca = use_rfca, all_rfca = all_rfca, mix_c = mix_c),
                    BasicBlock(w, w, use_rfca = use_rfca, all_rfca = all_rfca, mix_c = mix_c)
                )
            else:
                branch = nn.Sequential(
                    BasicBlock(w, w, use_rfca = use_rfca, all_rfca = all_rfca, mix_c = mix_c),
                    BasicBlock(w, w, use_rfca = use_rfca, all_rfca = all_rfca, mix_c = mix_c),
                    BasicBlock(w, w, use_rfca = use_rfca, all_rfca = all_rfca, mix_c = mix_c),
                    BasicBlock(w, w, use_rfca = use_rfca, all_rfca = all_rfca, mix_c = mix_c),
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
    def __init__(self, base_channel: int = 32, num_joints: int = 4, 
                 with_FFCA = True, spatial_attention = True, skip_connection = True, 
                 swap_att = False, use_rfca = True, all_rfca = True, mix_c = True):
        super().__init__()
        # Stem
        self.with_FFCA = with_FFCA
        self.skip_connection = skip_connection
        self.with_spacial = spatial_attention
        self.swap_att = swap_att
        self.mix_c = mix_c
        self.use_rfca = use_rfca
        self.mix_c = mix_c
        self.all_rfca = all_rfca
        #self.con11 = nn.Conv2d(base_channel, base_channel, kernel_size=1, stride=1, bias=False)
        #self.con12 = nn.Conv2d(base_channel*2, base_channel*2, kernel_size=1, stride=1, bias=False)
        #self.con13 = nn.Conv2d(base_channel*4, base_channel*4, kernel_size=1, stride=1, bias=False)
        #self.con14 = nn.Conv2d(base_channel*8, base_channel*8, kernel_size=1, stride=1, bias=False)
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
        if self.use_rfca & self.all_rfca:
            self.transition1 = nn.ModuleList([
                nn.Sequential(
                    RFCAConv(256, base_channel,3,1,mix_c=self.mix_c),
                    #nn.Conv2d(256, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Sequential(  # 这里又使用一次Sequential是为了适配原项目中提供的权重
                        RFCAConv(256, base_channel * 2,3,2,mix_c=self.mix_c),
                        #nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)
                    )
                )
            ])
        else:
            self.transition1 = nn.ModuleList([
                nn.Sequential(
                    #RFCAConv(256, base_channel,3,1),
                    nn.Conv2d(256, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Sequential(  # 这里又使用一次Sequential是为了适配原项目中提供的权重
                        #RFCAConv(256, base_channel * 2,3,2),
                        nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)
                    )
                )
            ])

        # Stage2
        self.stage2 = nn.Sequential(
            StageModule(input_branches=2, output_branches=2, c=base_channel,use_rfca = self.use_rfca, all_rfca = self.all_rfca, mix_c=self.mix_c),
        )
        if use_rfca & all_rfca:
        # transition2
            self.transition2 = nn.ModuleList([
                nn.Identity(),  # None,  - Used in place of "None" because it is callable
                nn.Identity(),  # None,  - Used in place of "None" because it is callable
                nn.Sequential(
                    nn.Sequential(
                        RFCAConv(base_channel * 2, base_channel * 4,3,2,mix_c=self.mix_c),
                        #nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)
                    )
                )
            ])
        else:
        # transition2
            self.transition2 = nn.ModuleList([
                nn.Identity(),  # None,  - Used in place of "None" because it is callable
                nn.Identity(),  # None,  - Used in place of "None" because it is callable
                nn.Sequential(
                    nn.Sequential(
                        #RFCAConv(base_channel * 2, base_channel * 4,3,2),
                        nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)
                    )
                )
            ])

        # Stage3
        self.stage3 = nn.Sequential(
            StageModule(input_branches=3, output_branches=3, c=base_channel,use_rfca = self.use_rfca, all_rfca = self.all_rfca, mix_c=self.mix_c),
            StageModule(input_branches=3, output_branches=3, c=base_channel,use_rfca = self.use_rfca, all_rfca = self.all_rfca, mix_c=self.mix_c),
            StageModule(input_branches=3, output_branches=3, c=base_channel,use_rfca = self.use_rfca, all_rfca = self.all_rfca, mix_c=self.mix_c),
            StageModule(input_branches=3, output_branches=3, c=base_channel,use_rfca=self.use_rfca, all_rfca = self.all_rfca, mix_c=self.mix_c)
        )
        if self.use_rfca & self.all_rfca:
        # transition3
            self.transition3 = nn.ModuleList([
                nn.Identity(),  # None,  - Used in place of "None" because it is callable
                nn.Identity(),  # None,  - Used in place of "None" because it is callable
                nn.Identity(),  # None,  - Used in place of "None" because it is callable
                nn.Sequential(
                    nn.Sequential(
                        RFCAConv(base_channel * 4, base_channel * 8,3,2,mix_c=self.mix_c),
                        #nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)
                    )
                )
            ])
        else:
           self.transition3 = nn.ModuleList([
                nn.Identity(),  # None,  - Used in place of "None" because it is callable
                nn.Identity(),  # None,  - Used in place of "None" because it is callable
                nn.Identity(),  # None,  - Used in place of "None" because it is callable
                nn.Sequential(
                    nn.Sequential(
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
            StageModule(input_branches=4, output_branches=4, c=base_channel,use_rfca=self.use_rfca, all_rfca = self.all_rfca, mix_c=self.mix_c),
            StageModule(input_branches=4, output_branches=4, c=base_channel,use_rfca=self.use_rfca, all_rfca = self.all_rfca, mix_c=self.mix_c),
            
            #将四个输入分支进行融合，输出一个分支
            #StageModule(input_branches=4, output_branches=1, c=base_channel)
        )

        if self.with_spacial & (not self.swap_att):
            self.spatial_attention = SPAtt()
            
        elif self.with_spacial & self.swap_att:
            self.cha_att1 = Channel_ATT(base_channel, self.swap_att)
            self.cha_att2 = Channel_ATT(base_channel*2, self.swap_att)
            self.cha_att3 = Channel_ATT(base_channel*4, self.swap_att)
            self.cha_att4 = Channel_ATT(base_channel*8, self.swap_att)
        if not self.with_FFCA:
            self.stage4.add_module("StageModule",StageModule(input_branches=4, output_branches=1, c=base_channel,use_rfca=False, all_rfca = False, mix_c=self.mix_c))
        else:
            self.decoder = myDecoder(base_channel,self.swap_att, self.use_rfca,self.mix_c)
        
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
        if self.skip_connection & self.with_spacial & (not self.swap_att): 
            #print("skipped")
            #skip_1 = skip_1 + x[0]
            skip_2 = skip_2 + x[1]
            skip_3 = skip_3 + x[2]
            skip_4 = skip_4 + x[3]
            x[0] = x[0] * self.spatial_attention(skip_1)
            x[1] = x[1] * self.spatial_attention(skip_2)
            x[2] = x[2] * self.spatial_attention(skip_3)
            x[3] = x[3] * self.spatial_attention(skip_4)
        elif self.skip_connection &  (not self.with_spacial):
            #x[0] = self.gelu(self.con11(x[0] + skip_1))
            #x[1] = self.gelu(self.con12(x[1] + skip_2))
            #x[2] = self.gelu(self.con13(x[2] + skip_3))
            #x[3] = self.gelu(self.con14(x[3] + skip_4))
            #print("skipped")
            #x[0] = skip_1 + x[0]
            x[1] = skip_2 + x[1]
            x[2] = skip_3 + x[2]
            x[3] = skip_4 + x[3]

        elif self.swap_att:
            #print("skipped")
            #skip_1 = skip_1 * x[0]
            skip_2 = skip_2 * x[1]
            skip_3 = skip_3 * x[2]
            skip_4 = skip_4 * x[3]
            x[0] = x[0] * self.cha_att1(skip_1)
            x[1] = x[1] * self.cha_att2(skip_2)
            x[2] = x[2] * self.cha_att3(skip_3)
            x[3] = x[3] * self.cha_att4(skip_4)
        if self.with_FFCA:
            x = self.decoder(x)
            x = self.final_layer(x)
        else:
        #由于最后一层只输出一个分支，所以这里只取最后一个分支x[0]
            x = self.final_layer(x[0])
        #x = self.sigmoid(x)

        return x



class myFFCA(nn.Module):
    """
    FFCA - Fusion Feature Channel Attention Module
    """
    def __init__(self, low_channel,high_channel,swap_att = False, use_rfca = True, mix_c =True):#输入为低级分支的通道个数
        super().__init__()
        #low_branch size should be (batch_size, 2*high_channel, height/2, width/2)

        self.swap_att = swap_att
        self.use_rfca = use_rfca
        self.low_channle = low_channel
        self.high_channle = high_channel
        self.sigmoid = nn.Sigmoid()
        #process input branch
        self.inbranch_process = nn.Sequential(
            #nn.ConvTranspose2d(self.low_channle,self.low_channle,kernel_size=4,stride=2,padding=1,bias=False),
            #upsample, make lowchannel's H,W double and equals to highchannel's H,W
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            
            #3*3conv, make lowchannel's channel equals to highchannel's channel
            nn.Conv2d(self.low_channle, self.high_channle,3,padding=1)
        )
        #Swap position for channel att and spatial att
        if self.swap_att:
            self.Spa_ATT = SPAtt()
        else:
            self.Channel_ATT = Channel_ATT(self.low_channle, self.swap_att)
        self.output = nn.Sequential(
            #3*3conv
            nn.Conv2d(self.low_channle, self.high_channle,3,padding=1),
            self.sigmoid
        )
        
        self.rfcaout = RFCAConv(self.high_channle*2, self.high_channle,3,1,mix_c=mix_c)
        #self.whfusion = whfusion(self.high_channle, self.high_channle,3,1)
        #self.mixedwh = whfusion(self.high_channle*2, self.high_channle,3,1)
        #self.conv1 = nn.Conv2d(2*self.high_channle, 2*self.high_channle,kernel_size=1,bias=False)
    def forward(self, low_branch, high_branch):
        up_lowbranch = self.inbranch_process(low_branch)
        #concatenate two branch
        concat_branch = torch.cat((up_lowbranch, high_branch), 1)#channle*2
        if self.swap_att:
            feature_weight = self.Spa_ATT(concat_branch)
        elif self.use_rfca:
            output_branch = self.rfcaout(concat_branch)
            #print(output_branch)
        else:
            feature_weight = self.Channel_ATT(concat_branch)

            weighted_branch = feature_weight * up_lowbranch
            output_branch = torch.cat((weighted_branch, high_branch), 1)#channle*2
            output_branch = self.output(output_branch)#half the channle	
            #print(output_branch.shape)
        return output_branch #output size should be (batch_size, high_channel, height, width)


class myDecoder(nn.Module):
    """
    Decoder with FFCA modules.
    """
    def __init__(self,basechannel,swap_att = False, use_rfca = True, mix_c = True):
        super().__init__()
        #from higher ffca to lower ffca
        self.ffca3 = myFFCA(2*basechannel,basechannel, swap_att, use_rfca, mix_c)
        self.ffca2 = myFFCA(4*basechannel,2*basechannel, swap_att, use_rfca, mix_c)
        self.ffca1 = myFFCA(8*basechannel,4*basechannel, swap_att, use_rfca, mix_c)
        
    def forward(self, x):
        branch3, branch2, branch1, branch0 = x
        ffca_output = self.ffca1(branch0, branch1)
        ffca_output = self.ffca2(ffca_output, branch2)
        ffca_output = self.ffca3(ffca_output, branch3)

        return ffca_output

class SPAtt(nn.Module):
    """
    Spatial Attention Module
    """
    def __init__(self):
        super().__init__()
        self.con1 = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(1, momentum=BN_MOMENTUM)
    #input is a tensor with size (batch_size, channel, height, width)
    #output is a tensor with size (batch_size, 1, height, width)
    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        #print("avg_out.shape: ", avg_out.shape)
        out = torch.cat([max_out, avg_out], dim=1)
        out = self.con1(out)

        out = self.bn(out)
        out = self.sigmoid(out)
        return out

class Channel_ATT(nn.Module):
    """
    Channel Attention Module
    """
    def __init__(self, low_channel, swap_att = False):
        super().__init__()
        self.low_channle = low_channel
        self.swap_att = swap_att
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        if not self.swap_att:
            self.fc = nn.Linear(self.low_channle, self.low_channle//2, bias=False)
        else:
            self.fc = nn.Linear(self.low_channle, self.low_channle, bias=False)

        self.softmax = nn.Softmax()
        
    def forward(self, x):
        maxpooled_branch = self.max_pool(x)
        avgpooled_branch = self.avg_pool(x)

        #change size from[batch_size, channel, 1, 1] to [batch_size, channel]
        maxpooled_branch_flat = maxpooled_branch.view(maxpooled_branch.size(0), -1)
        avgpooled_branch_flat = avgpooled_branch.view(avgpooled_branch.size(0), -1)
        # Pass the tensors through the fully connected layer
        feature_weight = self.softmax(self.fc(maxpooled_branch_flat) + self.fc(avgpooled_branch_flat))

        
        #restore the size of weight from[batch_size, channel] to [batch_size, channel, 1, 1]
        feature_weight = feature_weight.view(feature_weight.size(0), feature_weight.size(1), 1, 1)
        

        return feature_weight

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class RFCAConv(nn.Module):
    def __init__(self, inp, oup,kernel_size,stride, reduction=32, mix_c = True):
        super(RFCAConv, self).__init__()
        self.kernel_size = kernel_size
        self.mix_c = mix_c
        self.generate = nn.Sequential(nn.Conv2d(inp,inp * (kernel_size**2),kernel_size,padding=kernel_size//2,
                                                stride=stride,groups=inp,
                                                bias =False),
                                      nn.BatchNorm2d(inp * (kernel_size**2)),
                                      nn.ReLU()
                                      )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp,oup,kernel_size,stride=kernel_size))
        #self.spatt = SPAtt()
        self.conmaxmin = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        #self.mix_h = nn.Conv2d(inp, inp, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        #self.mix_w = nn.Conv2d(inp, inp, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3))

    def forward(self, x):
        b,c = x.shape[0:2]
        generate_feature = self.generate(x)
        h,w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b,c,self.kernel_size**2,h,w)
        
        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              n2=self.kernel_size)
        x_h = self.pool_h(generate_feature)
        #x_h_max = self.pool_h_max(generate_feature)
        x_w = self.pool_w(generate_feature).permute(0, 1, 3, 2)
        #x_w_max = self.pool_w_max(generate_feature).permute(0, 1, 3, 2)
        if self.mix_c:
            #print("mix C")
            x_c_mean = torch.mean(generate_feature, dim=1, keepdim=True)
            x_c_max = torch.max(generate_feature, dim=1, keepdim=True)[0]
            x_c = torch.cat([x_c_mean, x_c_max], dim=1)
            x_c = self.conmaxmin(x_c)
            #combine x_h and x_c
            x_h = x_h * self.pool_h(x_c)
            x_w = x_w * self.pool_w(x_c).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        h,w = generate_feature.shape[2:]
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()

        a_w = self.conv_w(x_w).sigmoid()

        out = self.conv(generate_feature * a_w * a_h)

        #out = out * self.spatt(out)
 
        return out