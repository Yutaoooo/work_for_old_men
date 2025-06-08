# BASNet 项目分析

## 文件结构说明
```
yuan/
├── model/                # 模型定义
│   ├── BASNet.py         # 主模型架构
│   │   ├── 编码器-解码器结构
│   │   │   - 编码器基于ResNet34
│   │   │   - 解码器包含多尺度特征融合
│   │   ├── Refine模块(RefUnet)
│   │   │   - 用于边界精细化
│   │   │   - 包含5层下采样和上采样结构
│   │   └── 多输出层
│   │       - 7个侧边输出+1个融合输出
│   │       - 每个输出层独立计算损失
│   └── resnet_model.py   # ResNet组件
│
├── basnet_train.py       # 训练脚本
│   ├── 数据准备
│   │   - 数据集：DUTS-TR
│   │   - 数据增强：Rescale/RandomCrop/ToTensor
│   ├── 损失函数
│   │   - BCE + SSIM + IOU组合损失
│   └── 训练参数
│       - Adam优化器(lr=0.001)
│       - Batch size=8
│       - Epoch=100000
│
├── basnet_test.py        # 测试脚本
│   ├── 输入处理
│   │   - Resize到256x256
│   │   - 转换为Tensor
│   ├── 推理过程
│   │   - 加载预训练模型
│   │   - 生成显著性图
│   └── 输出处理
│       - 保存为PNG格式
│       - 保持原始尺寸
│
├── data_loader.py        # 数据加载
├── test_data/            # 测试数据
│   ├── test_images/      # 输入图像
│   └── test_results/     # 输出结果
├── saved_models/         # 训练好的模型
├── figures/              # 结果示例图
└── README.md             # 项目说明
```

## 核心实现细节

### 编码器-解码器结构
**编码器部分**：
```python
self.inconv = nn.Conv2d(n_channels,64,3,padding=1)
self.encoder1 = resnet.layer1  # ResNet34 layer1
self.encoder2 = resnet.layer2  # layer2
self.encoder3 = resnet.layer3  # layer3
self.encoder4 = resnet.layer4  # layer4
```

**解码器部分**：
```python
self.conv4d_1 = nn.Conv2d(1024,512,3,padding=1)  # 特征融合
self.conv4d_2 = nn.Conv2d(512,256,3,padding=1)   # 通道减半
self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
```

### Refine模块
```python
# 下采样路径
self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

# 上采样路径
self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

# 残差连接
return x + residual
```

### 多输出层
```python
self.outconvb = nn.Conv2d(512,1,3,padding=1)  # 桥接层输出
self.outconv1 = nn.Conv2d(64,1,3,padding=1)   # 第1层输出
...
return F.sigmoid(dout), F.sigmoid(d1), ...  # 8个输出
```