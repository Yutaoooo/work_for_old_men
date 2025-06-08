import torch
from torch import nn
from torch.nn import init

"""
DSS网络实现文件
包含DSS网络的核心组件：
1. VGG基础网络
2. 特征提取层(FeatLayer)
3. 特征连接层(ConcatLayer) 
4. 特征融合层(FusionLayer)
5. 完整的DSS网络结构
"""

# 主干 VGG 网络的结构
base = {'dss': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

# 扩展层参数
extra = {'dss': [(64, 128, 3, [8, 16, 32, 64]), (128, 128, 3, [4, 8, 16, 32]), (256, 256, 5, [8, 16]),
                 (512, 256, 5, [4, 8]), (512, 512, 5, []), (512, 512, 7, [])]}
# 特征连接关系
connect = {'dss': [[2, 3, 4, 5], [2, 3, 4, 5], [4, 5], [4, 5], [], []]}


# 根据配置动态构建 VGG16 风格的卷积层序列，支持可选的 BatchNorm
def vgg(cfg, i=3, batch_norm=False):
    """
    构建VGG16基础网络
    参数:
        cfg: 网络配置列表，如[64,64,'M',128,128,'M',...]
        i: 输入通道数，默认为3(RGB)
        batch_norm: 是否使用批归一化
    返回:
        layers: 构建好的网络层列表
    """
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers

# 2. FeatLayer侧边输出层，对主干网络不同阶段的特征进行卷积处理，输出单通道特征图
class FeatLayer(nn.Module):
    def __init__(self, in_channel, channel, k):
        super(FeatLayer, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(in_channel, channel, k, 1, k // 2), nn.ReLU(inplace=True),
                                  nn.Conv2d(channel, channel, k, 1, k // 2), nn.ReLU(inplace=True),
                                  nn.Conv2d(channel, 1, 1, 1))

    def forward(self, x):
        return self.main(x)
    
# 3. ConcatLayer特征连接层，将主特征与多个辅助特征进行上采样、拼接和卷积融合，支持可选的上采样操作，便于多尺度特征对齐
class ConcatLayer(nn.Module):
    def __init__(self, list_k, k, scale=True):
        super(ConcatLayer, self).__init__()
        l, up, self.scale = len(list_k), [], scale
        for i in range(l):
            up.append(nn.ConvTranspose2d(1, 1, list_k[i], list_k[i] // 2, list_k[i] // 4))
        self.upconv = nn.ModuleList(up)
        self.conv = nn.Conv2d(l + 1, 1, 1, 1)
        self.deconv = nn.ConvTranspose2d(1, 1, k * 2, k, k // 2) if scale else None

    def forward(self, x, list_x):
        elem_x = [x]
        for i, elem in enumerate(list_x):
            elem_x.append(self.upconv[i](elem))
        if self.scale:
            out = self.deconv(self.conv(torch.cat(elem_x, dim=1)))
        else:
            out = self.conv(torch.cat(elem_x, dim=1))
        return out

# 4.特征融合层，可学习的特征加权融合，训练时会自动调整各分支特征的贡献比例
class FusionLayer(nn.Module):
    def __init__(self, nums=6):
        super(FusionLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(nums))
        self.nums = nums
        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.weights, 1 / self.nums)

    def forward(self, x):
        for i in range(self.nums):
            out = self.weights[i] * x[i] if i == 0 else out + self.weights[i] * x[i]
        return out


# 根据扩展配置生成一组 FeatLayer 和 ConcatLayer，用于多分支特征处理和融合
def extra_layer(vgg, cfg):
    feat_layers, concat_layers, scale = [], [], 1
    for k, v in enumerate(cfg):
        # side output (paper: figure 3)
        feat_layers += [FeatLayer(v[0], v[1], v[2])]
        # feature map before sigmoid
        concat_layers += [ConcatLayer(v[3], scale, k != 0)]
        scale *= 2
    return vgg, feat_layers, concat_layers


# DSS网络
# 整个网络的封装，负责特征提取、分支处理、特征融合和最终输出
class DSS(nn.Module):
    def __init__(self, base, feat_layers, concat_layers, connect, extract=[3, 8, 15, 22, 29], v2=True):
        super(DSS, self).__init__()
        self.extract = extract
        self.connect = connect
        self.base = nn.ModuleList(base)
        self.feat = nn.ModuleList(feat_layers)
        self.comb = nn.ModuleList(concat_layers)
        self.pool = nn.AvgPool2d(3, 1, 1)
        self.v2 = v2
        if v2: self.fuse = FusionLayer()

    # 通过主干网络提取多层特征，然后经过各自的 FeatLayer 和 ConcatLayer 处理，
    # 最后通过 FusionLayer（或简单均值）融合所有分支，输出经过 sigmoid 激活的概率图
    def forward(self, x, label=None):
        prob, back, y, num = list(), list(), list(), 0
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.extract:
                y.append(self.feat[num](x))
                num += 1
        # side output
        y.append(self.feat[num](self.pool(x)))
        for i, k in enumerate(range(len(y))):
            back.append(self.comb[i](y[i], [y[j] for j in self.connect[i]]))
        # fusion map
        if self.v2:
            # version2: learning fusion
            back.append(self.fuse(back))
        else:
            # version1: mean fusion
            back.append(torch.cat(back, dim=1).mean(dim=1, keepdim=True))
        # add sigmoid
        for i in back: prob.append(torch.sigmoid(i))
        return prob


# build the whole network
def build_model():
    return DSS(*extra_layer(vgg(base['dss'], 3), extra['dss']), connect['dss'])


# 权重初始化
def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


if __name__ == '__main__':
    net = build_model()
    img = torch.randn(1, 3, 64, 64)
    net = net.to(torch.device('cuda:0'))
    img = img.to(torch.device('cuda:0'))
    out = net(img)
    k = [out[x] for x in [1, 2, 3, 6]]
    print(len(k))
    # for param in net.parameters():
    #     print(param)


    """显著目标检测的深侧输出监督网络（DSS）。
    本类实现了本文所描述的决策支持系统网络体系结构
    Hou等人的“短连接深度监督显著目标检测”。
    该网络由基础网络（通常为VGG）、特征提取层、
    用于侧输出的连接层，以及层之间的连接。

    参数:
    base (list)：基网络层列表
    feat_layers (list)：特征提取层列表
    concat_layers (list)：侧输出的连接层列表
    connect (list)：指定层连接的列表
    extract （list，可选）：要提取特征的层索引。默认为[3,8,15,22,29]
    v2 （bool，可选）：是否使用可学习融合的版本2。默认为True

    属性:
    extract (list)：要提取特征的层索引
    connect (list)：层连接规范
    base (n . modulelist)：基础网络层
    feat (n . modulelist)：特征提取层
    comb (n . modulelist)：连接层
    pool (n . avgpool2d)：平均池化层
    v2 (bool)：版本标志
    fuse (FusionLayer): v2的融合层

    返回:
    list：概率图列表，包括侧输出和融合输出
    """