import os
import torch
from torchvision import models
from torchvision.models import VGG16_Weights

# extract vgg features
if __name__ == '__main__':
    save_fold = '../weights'
    if not os.path.exists(save_fold):
        os.mkdir(save_fold)
    
    # 加载 VGG16 模型并使用预训练权重
    vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    
    # 保存 VGG16 的特征提取部分的权重
    torch.save(vgg.features.state_dict(), os.path.join(save_fold, 'vgg16_feat.pth'))