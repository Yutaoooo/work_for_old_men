import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import BASNet

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output(image_name, pred, d_dir):
    import os
    from PIL import Image
    import numpy as np
    from skimage import io

    if not os.path.exists(d_dir):
        os.makedirs(d_dir)
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = os.path.basename(image_name)  # 只取文件名（如 0003.jpg）
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    imidx = os.path.splitext(img_name)[0]    # 去掉扩展名（如 0003）
    save_path = os.path.join(d_dir, imidx + '.png')
    imo.save(save_path)



if __name__ == '__main__':
	# --------- 1. get image path and name ---------
	
	image_dir = './test_data/test_images/'
	prediction_dir = './test_data/test_results/'
	model_dir = './saved_models/basnet_bsi/basnet.pth'
	
	img_name_list = glob.glob(image_dir + '*.jpg')
	
	# --------- 2. dataloader ---------
	#1. dataload
	test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [],transform=transforms.Compose([RescaleT(256),ToTensorLab(flag=0)]))
	test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,shuffle=False,num_workers=1)
	
	# --------- 3. model define ---------
	print("...load BASNet...")
	net = BASNet(3,1)
	net.load_state_dict(torch.load(model_dir))
	if torch.cuda.is_available():
		net.cuda()
	net.eval()
	
	# --------- 4. inference for each image ---------
	for i_test, data_test in enumerate(test_salobj_dataloader):
	
		print("inferencing:",img_name_list[i_test].split("/")[-1])
	
		inputs_test = data_test['image']
		inputs_test = inputs_test.type(torch.FloatTensor)
	
		if torch.cuda.is_available():
			inputs_test = Variable(inputs_test.cuda())
		else:
			inputs_test = Variable(inputs_test)
	
		d1,d2,d3,d4,d5,d6,d7,d8 = net(inputs_test)
	
		# normalization
		pred = d1[:,0,:,:]
		pred = normPRED(pred)
	
		# save results to test_results folder
		save_output(img_name_list[i_test],pred,prediction_dir)
	
		del d1,d2,d3,d4,d5,d6,d7,d8
