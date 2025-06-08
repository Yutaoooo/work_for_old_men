import torch
import torch.nn.functional as F
import numpy as np
import pdb, os, argparse
from datetime import datetime
from model.vgg_models import Back_VGG
from data import get_loader
from utils import clip_gradient, adjust_lr
import os
from PIL import Image  # 导入PIL库
import smoothness
import matplotlib.pyplot as plt

def visualize_prediction1(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_sal1.png'.format(kk)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img = Image.fromarray(pred_edge_kk)  # 将numpy数组转换为Pillow图像对象
        img.save(save_path + name)  # 保存图像

def visualize_prediction2(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_sal2.png'.format(kk)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img = Image.fromarray(pred_edge_kk)  # 将numpy数组转换为Pillow图像对象
        img.save(save_path + name)  # 保存图像

def visualize_edge(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_edge.png'.format(kk)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img = Image.fromarray(pred_edge_kk)  # 将numpy数组转换为Pillow图像对象
        img.save(save_path + name)  # 保存图像

def train(train_loader, model, optimizer, epoch, opt, total_step, CE, smooth_loss):
    model.train()
    sal1_losses = []
    edge_losses = []
    sal2_losses = []
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts, masks, grays, edges = pack
        images = images.cuda()
        gts = gts.cuda()
        masks = masks.cuda()
        grays = grays.cuda()
        edges = edges.cuda()

        img_size = images.size(2) * images.size(3) * images.size(0)
        ratio = img_size / torch.sum(masks)

        sal1, edge_map, sal2 = model(images)

        sal1_prob = torch.sigmoid(sal1)
        sal1_prob = sal1_prob * masks
        sal2_prob = torch.sigmoid(sal2)
        sal2_prob = sal2_prob * masks

        #移除smooth_loss
        #smoothLoss_cur1 = opt.sm_loss_weight * smooth_loss(torch.sigmoid(sal1), grays)
        sal_loss1 = ratio * CE(sal1_prob, gts * masks) #+ smoothLoss_cur1
        # 移除smooth_loss
        #smoothLoss_cur2 = opt.sm_loss_weight * smooth_loss(torch.sigmoid(sal2), grays)
        sal_loss2 = ratio * CE(sal2_prob, gts * masks) #+ smoothLoss_cur2
        edge_loss = opt.edge_loss_weight * CE(torch.sigmoid(edge_map), edges)
        #移除smooth_loss
        bce = sal_loss1 + edge_loss + sal_loss2
        visualize_prediction1(torch.sigmoid(sal1))
        visualize_edge(torch.sigmoid(edge_map))
        visualize_prediction2(torch.sigmoid(sal2))

        loss = bce
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        sal1_losses.append(sal_loss1.item())
        edge_losses.append(edge_loss.item())
        sal2_losses.append(sal_loss2.item())

        if i % 1 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], sal1_loss: {:0.4f}, edge_loss: {:0.4f}, sal2_loss: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, sal_loss1.item(), edge_loss.item(), sal_loss2.item()))

    save_path = 'models/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + 'scribble' + '_%d' % epoch + '.pth')

    return sal1_losses, edge_losses, sal2_losses

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 设置使用GPU 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=20, help='every n epochs decay learning rate')
    parser.add_argument('--sm_loss_weight', type=float, default=0.3, help='weight for smoothness loss')
    parser.add_argument('--edge_loss_weight', type=float, default=1.0, help='weight for edge loss')
    opt = parser.parse_args()

    print('Learning Rate: {}'.format(opt.lr))
    # build models
    model = Back_VGG(channel=32)

    model.cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    # 更新数据路径
    image_root = './data/data/img/'
    gt_root = './data/data/gt/'
    mask_root = './data/data/mask/'
    edge_root = './data/data/edge/'
    grayimg_root = './data/data/gray/'
    train_loader = get_loader(image_root, gt_root, mask_root, grayimg_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    CE = torch.nn.BCELoss()
    #smooth_loss = smoothness.smoothness_loss(size_average=True) #移除smooth_loss

    all_sal1_losses = []
    all_edge_losses = []
    all_sal2_losses = []

    print("Scribble it!")
    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        sal1_losses, edge_losses, sal2_losses = train(train_loader, model, optimizer, epoch, opt, total_step, CE, smooth_loss)
        all_sal1_losses.extend(sal1_losses)
        all_edge_losses.extend(edge_losses)
        all_sal2_losses.extend(sal2_losses)

    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(all_sal1_losses, label='Sal1 Loss')
    plt.plot(all_edge_losses, label='Edge Loss')
    plt.plot(all_sal2_losses, label='Sal2 Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.savefig('training_losses.png')
    plt.show()