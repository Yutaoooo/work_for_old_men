import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
from PIL import Image  # 导入 PIL 库

from model.vgg_models import Back_VGG
from data import test_dataset
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
opt = parser.parse_args()

dataset_path = './testing/img/'

model = Back_VGG(channel=32)
# 加载保存的状态字典
checkpoint = torch.load('./models/scribbleWithoutSmooth_30.pth')
# 检查是否包含 'model_state_dict' 键
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# 过滤掉意外的键
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

# 加载过滤后的状态字典
model.load_state_dict(filtered_state_dict)

model.cuda()
model.eval()

test_datasets = ['ECSSD', 'DUT', 'DUTS_Test', 'THUR', 'HKU-IS']

for dataset in test_datasets:
    save_path = './results/ResNet50withoutSmooth/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/'
    test_loader = test_dataset(image_root, opt.testsize)
    for i in range(test_loader.size):
        print(i)
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        res0, res1, res = model(image)
        # 使用 F.interpolate 替代 F.upsample
        res = F.interpolate(res, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # 将 res 转换为 0-255 的整数类型
        res = (res * 255).astype(np.uint8)
        # 使用 PIL 库保存图像
        img = Image.fromarray(res)
        img.save(save_path + name)