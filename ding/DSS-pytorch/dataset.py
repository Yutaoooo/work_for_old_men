import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms


class ImageData(data.Dataset):
    """ image dataset
    img_root:    image root (root which contain images)
    label_root:  label root (root which contains labels)
    transform:   pre-process for image
    t_transform: pre-process for label
    filename:    MSRA-B use xxx.txt to recognize train-val-test data (only for MSRA-B)
    """
# image_path = 'D:/d_CODE/github/MSRA-B/image'
# label_path = 'D:/d_CODE/github/MSRA-B/annotation'
# train_file = 'D:/d_CODE/github/MSRA-B/train_cvpr2013.txt'
# valid_file = 'D:/d_CODE/github/MSRA-B/valid_cvpr2013.txt'
# test_file = 'D:/d_CODE/github/MSRA-B/test_cvpr2013.txt'

# 第一项默认是照片路径
# 第二项默认是标签路径
# 第三项默认是256
# 第四项默认是batch 为 1
# 第五项默认是训练文件名字集合
# 第六项默认是线程数 为 4 
# train_loader = get_loader(config.train_path, config.label_path, config.img_size, config.batch_size,
#                           filename=config.train_file, num_thread=config.num_thread)
    def __init__(self, img_root, label_root, transform, t_transform, filename=None):
        if filename is None:
    # 如果没有提供文件名，则默认加载 img_root 目录下的所有文件
    # 假设 img_root = 'D:/d_CODE/github/MSRA-B/image'
    # 目录内容为 ['1.jpg', '2.jpg', '3.jpg']
    # 则 self.image_path = ['D:/d_CODE/github/MSRA-B/image/1.jpg', 
    #                       'D:/d_CODE/github/MSRA-B/image/2.jpg', 
    #                       'D:/d_CODE/github/MSRA-B/image/3.jpg']
            self.image_path = [os.path.join(img_root, fname) for fname in os.listdir(img_root)]
    
    # 对应的标签路径假设 label_root = 'D:/d_CODE/github/MSRA-B/annotation'
    # 则 self.label_path = ['D:/d_CODE/github/MSRA-B/annotation/1.png', 
    #                        'D:/d_CODE/github/MSRA-B/annotation/2.png', 
    #                        'D:/d_CODE/github/MSRA-B/annotation/3.png']
            self.label_path = [
                os.path.join(label_root, os.path.splitext(os.path.basename(fname))[0] + '.png')
                for fname in self.image_path
            ]
        else:
    # 如果提供了文件名，则从文件中读取每一行的内容作为文件名
    # 假设 filename = 'D:/d_CODE/github/MSRA-B/train_cvpr2013.txt'
    # 文件内容为：
    # 1
    # 2
    # 3
    # 则 lines = ['1', '2', '3']
            with open(filename, 'r') as file:
                lines = [line.strip() for line in file]
    # 读取文件名时去掉空格和换行符    
    # 根据 lines 中的文件名生成图像路径
    # 假设 img_root = 'D:/d_CODE/github/MSRA-B/image'
    # 则 self.image_path = ['D:/d_CODE/github/MSRA-B/image/1.jpg', 
    #                       'D:/d_CODE/github/MSRA-B/image/2.jpg', 
    #                       'D:/d_CODE/github/MSRA-B/image/3.jpg']
            self.image_path = [os.path.join(img_root, line + '.jpg') for line in lines]
    
    # 根据 lines 中的文件名生成标签路径
    # 假设 label_root = 'D:/d_CODE/github/MSRA-B/annotation'
    # 则 self.label_path = ['D:/d_CODE/github/MSRA-B/annotation/1.png', 
    #                        'D:/d_CODE/github/MSRA-B/annotation/2.png', 
    #                        'D:/d_CODE/github/MSRA-B/annotation/3.png']
            self.label_path = [os.path.join(label_root, line + '.png') for line in lines]

            self.transform = transform
            self.t_transform = t_transform

    def __getitem__(self, item):
        image = Image.open(self.image_path[item])
        label = Image.open(self.label_path[item]).convert('L')
        if self.transform is not None:
            image = self.transform(image)
        if self.t_transform is not None:
            label = self.t_transform(label)
        return image, label

    def __len__(self):
        return len(self.image_path)


# get the dataloader (Note: without data augmentation)
def get_loader(img_root, label_root, img_size, batch_size, filename=None, mode='train', num_thread=4, pin=True):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        t_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(round_tensor)  # TODO: it maybe unnecessary
        ])
        dataset = ImageData(img_root, label_root, transform, t_transform, filename=filename)
        data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_thread,
                                      pin_memory=pin)
        return data_loader
    else:
        t_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
        dataset = ImageData(img_root, label_root, None, t_transform, filename=filename)
        return dataset


def round_tensor(x):
    return torch.round(x)

if __name__ == '__main__':
    import numpy as np
    img_root = 'D:/d_CODE/github/MSRA-B/image'
    label_root = 'D:/d_CODE/github/MSRA-B/annotation'
    filename = 'D:/d_CODE/github/MSRA-B/train_cvpr2013.txt'
    loader = get_loader(img_root, label_root, 224, 1, filename=filename, mode='test')
    for image, label in loader:
        print(np.array(image).shape)
        break
