import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
BN_MOMENTUM = 0.1

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_rfca = True, mix_c =True):
        super(BasicBlock, self).__init__()
        if use_rfca:
            self.conv1 = RFCAConv(inplanes, planes,3,stride,mix_c=mix_c)
            #self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
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
    def __init__(self, input_branches, output_branches, c, use_rfca = True, mix_c = True):
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
                    BasicBlock(w, w, use_rfca = False,  mix_c = mix_c),
                    BasicBlock(w, w, use_rfca = False, mix_c = mix_c),
                    BasicBlock(w, w, use_rfca = False, mix_c = mix_c),
                    BasicBlock(w, w, use_rfca = False, mix_c = mix_c),
                )
            else:
                branch = nn.Sequential(
                    BasicBlock(w, w, use_rfca = use_rfca, mix_c = mix_c),
                    BasicBlock(w, w, use_rfca = use_rfca, mix_c = mix_c),
                    BasicBlock(w, w, use_rfca = use_rfca, mix_c = mix_c),
                    BasicBlock(w, w, use_rfca = use_rfca, mix_c = mix_c),
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

class PagFM(nn.Module):
    def __init__(self, low_channels, high_channels, after_relu=True, with_channel=True, BatchNorm=nn.BatchNorm2d, my_fusion = True, mix_c = True):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.my_fusion = my_fusion
        #x:high channel,y:low_channel
        #self.conv1 = nn.Conv2d(low_channels,high_channels,kernel_size=3,padding=0)
    

        if my_fusion:
            self.att = CA_module(high_channels,kernel_size=3,mix_c=mix_c)
            self.f_x = nn.Sequential(
                                    feature_gen(high_channels,high_channels,kernel_size=3,stride=1),
                                    BatchNorm(high_channels)
                                    )
            self.f_y = nn.Sequential(
                                    feature_gen(low_channels,high_channels,kernel_size=3,stride=1),
                                    
                                    BatchNorm(high_channels)
                                    )
        else:
            self.f_x = nn.Sequential(
                                    nn.Conv2d(high_channels, high_channels, 
                                              kernel_size=1, bias=False),
                                    BatchNorm(high_channels)
                                    )
            self.f_y = nn.Sequential(
                                    nn.Conv2d(high_channels, high_channels, 
                                              kernel_size=1, bias=False),
                                    BatchNorm(high_channels)
                                    )
        if with_channel:
            if my_fusion:
                self.Channel_ATT = Channel_ATT(high_channels)
                self.up = nn.Sequential(
                                        feature_gen(high_channels,high_channels,kernel_size=1,stride=1),
                                        #nn.Conv2d(high_channels,high_channels, 
                                                #kernel_size=1, bias=False),
                                        BatchNorm(high_channels)
                                    )
            else:
                self.up = nn.Sequential(
                                        nn.Conv2d(high_channels,high_channels, 
                                                kernel_size=1, bias=False),
                                        BatchNorm(high_channels)
                                    )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, y):
        #x is high,y is low
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        #生成特征图
        if self.my_fusion:
            #NOTE：DO NOT USE "Bilinear Mode" OR THE RESULTS WONT BE REPRODUCED
            y = nn.Upsample(size=[input_size[2], input_size[3]], mode='nearest')(y)
            y_q = self.f_y(y)
        else:
            y_q = self.f_y(y)
            y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='nearest')
        
        #上采样插值,y_q形状=x_k
        x_k = self.f_x(x)
        if self.my_fusion:
            input_size = x_k.size()
        
        if self.with_channel:
            sim_map = torch.sigmoid(self.Channel_ATT(self.up(x_k * y_q))*(self.up(x_k * y_q)))#sim map越大，说明两个特征图越相似
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        if self.my_fusion:
            x = sim_map*x_k + (1-sim_map)*y_q
            x = self.att(x)
        else:
            y = F.interpolate(y, size=[input_size[2], input_size[3]],
                                mode='bilinear', align_corners=False)
            x = (1-sim_map)*x + sim_map*y
        
        return x

class myFFCA(nn.Module):
    """
    FFCA - Fusion Feature Channel Attention Module
    """
    def __init__(self, low_channel,high_channel,
                 use_rfca = True, mix_c =True,
                   pag_fusion = True, my_fusion = True):#输入为低级分支的通道个数
        super().__init__()
        #low_branch size should be (batch_size, 2*high_channel, height/2, width/2)

        self.use_rfca = use_rfca
        self.my_fusion = my_fusion
        self.low_channle = low_channel
        self.high_channle = high_channel
        self.sigmoid = nn.Sigmoid()
        self.pag_fusion = pag_fusion
       
        #process input branch

        if pag_fusion:
            self.PagFM = PagFM(self.low_channle, self.high_channle, after_relu = False, with_channel = True, my_fusion = my_fusion, mix_c = mix_c)
            if not my_fusion:
                self.channel_reduce = RFCAConv(self.low_channle, self.high_channle,3,1,mix_c=mix_c)
                self.high_branch = RFCAConv(self.high_channle, self.high_channle,3,1,mix_c=mix_c)
        else:
            self.inbranch_process = nn.Sequential(
                #nn.ConvTranspose2d(self.low_channle,self.low_channle,kernel_size=4,stride=2,padding=1,bias=False),
                #upsample, make lowchannel's H,W double and equals to highchannel's H,W
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                
                #3*3conv, make lowchannel's channel equals to highchannel's channel
                nn.Conv2d(self.low_channle, self.high_channle,3,padding=1)
            )
            if use_rfca:
                self.rfcaout = RFCAConv(self.high_channle*2, self.high_channle,3,1,mix_c=mix_c)
            else:
            
                self.Channel_ATT = Channel_ATT(self.low_channle)
                self.output = nn.Sequential(
                    #3*3conv
                    nn.Conv2d(self.low_channle, self.high_channle,3,padding=1),
                    self.sigmoid
                )
    def forward(self, low_branch, high_branch):
        
        #concatenate two branch

        if self.pag_fusion:
            if self.my_fusion:
                output_branch = self.PagFM(high_branch, low_branch)
            else:

                up_lowbranch = self.channel_reduce(low_branch)
                high_branch = self.high_branch(high_branch)
                output_branch = self.PagFM(high_branch, up_lowbranch)
                #output_branch = self.rfcaout(output_branch)
        else:
            up_lowbranch = self.inbranch_process(low_branch)
            concat_branch = torch.cat((up_lowbranch, high_branch), 1)
            if self.use_rfca:
                output_branch = self.rfcaout(concat_branch)
            else:
                feature_weight = self.Channel_ATT(concat_branch)
                weighted_branch = feature_weight * up_lowbranch
                output_branch = torch.cat((weighted_branch, high_branch), 1)
                output_branch = self.output(output_branch)
        return output_branch #output size should be (batch_size, high_channel, height, width)


class myDecoder(nn.Module):
    """
    Decoder with FFCA modules.
    """
    def __init__(self,basechannel,
                 use_rfca = True, mix_c = True,
                 pag_fusion = True, my_fusion = True):
        super().__init__()
        #from higher ffca to lower ffca
        self.ffca3 = myFFCA(2*basechannel,basechannel,  use_rfca, mix_c, pag_fusion, my_fusion)
        self.ffca2 = myFFCA(4*basechannel,2*basechannel, use_rfca, mix_c, pag_fusion, my_fusion)
        self.ffca1 = myFFCA(8*basechannel,4*basechannel, use_rfca, mix_c, pag_fusion, my_fusion)
        
    def forward(self, x):
        branch3, branch2, branch1, branch0 = x
        ffca_output = self.ffca1(branch0, branch1)
        ffca_output = self.ffca2(ffca_output, branch2)
        ffca_output = self.ffca3(ffca_output, branch3)

        return ffca_output


class Channel_ATT(nn.Module):
    """
    Channel Attention Module
    """
    def __init__(self, low_channel):
        super().__init__()
        self.low_channle = low_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(self.low_channle, self.low_channle, bias=False)

        self.softmax = nn.Softmax()
        
    def forward(self, x):
        #_,_,h,w = x.size()
        #NOTE: DO NOT USE ADAPTIVE POOL OR THE RESULTS WONT BE REPRODUCED
        #max_pool = nn.MaxPool2d((h, w), stride=(h, w))
        #avg_pool = nn.AvgPool2d((h, w), stride=(h, w))
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
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp,oup,kernel_size,stride=kernel_size))
        #self.spatt = SPAtt()
        if mix_c:
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
                              
        #_, _, h, w = generate_feature.size()
        #NOTE: DO NOT USE ADAPTIVE POOL OR THE RESULTS WONT BE REPRODUCED
        x_h = self.pool_h(generate_feature)
        x_w = self.pool_w(generate_feature).permute(0, 1, 3, 2)
        #avg_pool_h = nn.AvgPool2d((1, w), stride=(1, w))
        #x_h = avg_pool_h(generate_feature)
        #avg_pool_w = nn.AvgPool2d((h, 1), stride=(h, 1))
        #x_w = avg_pool_w(generate_feature).permute(0, 1, 3, 2)
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

        out = self.conv(generate_feature * a_w * a_h )
        #out = out * self.spatt(out)
        return out
class feature_gen(nn.Module):
    def __init__(self, inp,mip,kernel_size,stride):
        super(feature_gen, self).__init__()
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(nn.Conv2d(inp,inp * (kernel_size**2),kernel_size,padding=kernel_size//2,
                                                stride=stride,groups=inp,
                                                bias =False),
                                        nn.BatchNorm2d(inp * (kernel_size**2)),
                                        nn.ReLU()
                                        )
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
    def forward(self,x):
        b,c = x.shape[0:2]
        generate_feature = self.generate(x)
        h,w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b,c,self.kernel_size**2,h,w)
        
        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              n2=self.kernel_size)
        generate_feature = self.conv1(generate_feature)
        return generate_feature
    
class CA_module(nn.Module):
    def __init__(self,inp,kernel_size =1,mix_c = True):
        super(CA_module, self).__init__()
        self.mix_c = mix_c
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_h = nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp,inp,kernel_size,stride=kernel_size))
        self.conv1 = nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(inp)
        self.act = h_swish()
        #self.spatt = SPAtt()
        self.conmaxmin = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
    def forward(self,generate_feature):
        #_, _, h, w = generate_feature.size()
        #NOTE: DO NOT USE ADAPTIVE POOL OR THE RESULTS WONT BE REPRODUCED
        x_h = self.pool_h(generate_feature)
        x_w = self.pool_w(generate_feature).permute(0, 1, 3, 2)
        
        
        #x_h_max = self.pool_h_max(generate_feature)
        #x_w = self.pool_w(generate_feature).permute(0, 1, 3, 2)
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

        out = self.conv(generate_feature * a_w * a_h )
        return out