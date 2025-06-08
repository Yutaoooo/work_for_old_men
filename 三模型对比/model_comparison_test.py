import os
import torch
import numpy as np
from skimage import io
from PIL import Image
import time
import json
from torchvision import transforms
from tqdm import tqdm

# 评估指标计算
def compute_metrics(pred, gt):
    pred = pred > 0.5
    gt = gt > 0.5
    
    tp = np.sum(np.logical_and(pred, gt))
    fp = np.sum(np.logical_and(pred, np.logical_not(gt)))
    fn = np.sum(np.logical_and(np.logical_not(pred), gt))
    tn = np.sum(np.logical_and(np.logical_not(pred), np.logical_not(gt)))
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    iou = tp / (tp + fp + fn + 1e-10)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'iou': float(iou)
    }

# 测试函数
def test_model(model, data_loader, device):
    model.eval()
    metrics = []
    inference_times = []
    
    with torch.no_grad():
        for data in tqdm(data_loader):
            inputs = data['image'].to(device)
            labels = data['label'].to(device)
            
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            
            if isinstance(outputs, tuple):
                pred = outputs[0]  # 取主输出
            else:
                pred = outputs
                
            pred = pred.squeeze().cpu().numpy()
            gt = labels.squeeze().cpu().numpy()
            
            metrics.append(compute_metrics(pred, gt))
            inference_times.append(end_time - start_time)
    
    # 计算平均指标
    avg_metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0].keys()}
    avg_metrics['inference_time'] = np.mean(inference_times)
    
    return avg_metrics

# 测试Ding模型
def test_ding_model():
    from ding.DSS_pytorch.dssnet import DSSNet
    from ding.DSS_pytorch.dataset import SalObjDataset
    from ding.DSS_pytorch.data_loader import RescaleT, ToTensorLab
    from torch.utils.data import DataLoader
    import glob
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DSSNet().to(device)
    model.load_state_dict(torch.load('ding/DSS-pytorch/results/run-10/models/final.pth'))
    
    test_dataset = SalObjDataset(
        img_name_list=glob.glob('ding/DSS-pytorch/test_images/*.jpg'),
        lbl_name_list=glob.glob('ding/DSS-pytorch/test_gt/*.png'),
        transform=transforms.Compose([RescaleT(256), ToTensorLab(flag=0)]))
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return test_model(model, test_loader, device)

# 测试Yuan模型  
def test_yuan_model():
    from yuan.model.BASNet import BASNet
    from yuan.data_loader import SalObjDataset
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BASNet(3,1).to(device)
    model.load_state_dict(torch.load('yuan/saved_models/basnet_bsi/basnet.pth'))
    
    test_dataset = SalObjDataset(
        img_name_list=glob.glob('ding/DSS-pytorch/test_images/*.jpg'),
        lbl_name_list=glob.glob('ding/DSS-pytorch/test_gt/*.png'), 
        transform=transforms.Compose([RescaleT(256), ToTensorLab(flag=0)]))
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return test_model(model, test_loader, device)

# 测试Tao模型
def test_tao_model():
    from Tao.DMT.model import DMTNet
    from Tao.DMT.dataloader import SODDataset
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DMTNet().to(device)
    model.load_state_dict(torch.load('Tao/DMT/saved_models/final.pth'))
    
    test_dataset = SODDataset(
        img_dir='ding/DSS-pytorch/test_images',
        gt_dir='ding/DSS-pytorch/test_gt',
        transform=transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()]))
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return test_model(model, test_loader, device)

# 主函数
if __name__ == '__main__':
    results = {
        'Ding': test_ding_model(),
        'Yuan': test_yuan_model(),
        'Tao': test_tao_model()
    }
    
    print("\n模型性能对比结果:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}模型:")
        print(f"平均推理时间: {metrics['inference_time']:.4f}s")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}") 
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"IoU: {metrics['iou']:.4f}")
    
    # 保存结果到JSON文件
    with open('model_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n测试完成，结果已保存到model_comparison_results.json")
