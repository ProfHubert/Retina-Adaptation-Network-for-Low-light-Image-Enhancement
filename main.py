import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
from metrics import psnr, ssim
from models import *
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn

import torchvision.utils as vutils
warnings.filterwarnings('ignore')
from option import opt,model_name,log_dir
from data_utils import *
from torchvision.models import vgg16
from PIL import Image
import random
from piqa import SSIM
from rgb2lab import *
import torch.nn.functional as fu

from torchvision import transforms
toPIL = transforms.ToPILImage()
itrs = np.concatenate((np.arange(0, 500, 50), np.arange(0, 3000, 100), np.arange(0, 10000, 1000), np.arange(0, 100000, 1000)), axis=0)


class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)

L_ssim = SSIMLoss().to('cuda')
Pce_loss = VGGPerceptualLoss()
convertor = RGB_HSV()
models_={
 	'bst':BST(gps=opt.gps,blocks=opt.blocks), 
}

loaders_={
	'its_train':ITS_train_loader,
	'its_test':ITS_test_loader,
	'ots_train':OTS_train_loader,
	'ots_test':OTS_test_loader
}

start_time=time.time()
T=opt.steps	
def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr

def train(net,loader_train,loader_test,optim,criterion):
	losses=[]
	start_step=0
	max_ssim=0
	max_psnr=0
	ssims=[]
	psnrs=[]
	mean_filter = torch.FloatTensor([[1,1,1],[1,-8,1],[1,1,1]]).cuda().unsqueeze(0).unsqueeze(0)
	edeg_eval = nn.L1Loss().to('cuda')
    
	if opt.resume and os.path.exists(opt.model_dir):
		print(f'resume from {opt.model_dir}')
		ckp=torch.load(opt.model_dir)
		losses=ckp['losses']
		start_step=ckp['step']
		max_ssim=ckp['max_ssim']
		max_psnr=ckp['max_psnr']
		psnrs=ckp['psnrs']
		ssims=ckp['ssims']
		print(f'start_step:{start_step} start training ---')
	else :
		print('train from scratch *** ')
      
	start_step=0
	eval_step=2000
	net = net.module
	for step in range(start_step+1,opt.steps+1):
		net.train()
		lr=opt.lr
		if not opt.no_lr_sche:
			lr=lr_schedule_cosdecay(step,T)
			for param_group in optim.param_groups:
				param_group["lr"] = lr  
		x, y=next(iter(loader_train))
		x=x.to(opt.device);y=y.to(opt.device);
		out, gph = net(x) 
		loss1 = criterion[0](out, y) + L_ssim(out, y) 
		loss2 = criterion[0](gph, y) 
		
        # lab color loss
		lab_gt = xyz2lab(rgb2xyz(y)) 
		lab_pred = xyz2lab(rgb2xyz(out)) 
		ab_gt, ab_pred = lab_gt[:, 1:, :, :] / 100, lab_pred[:, 1:, :, :] / 100  
		color_loss = criterion[0](ab_pred, ab_gt) + criterion[0](torch.pow(torch.pow(lab_gt[:, 1, :, :], 2) + torch.pow(lab_gt[:, 2, :, :], 2), 0.5).unsqueeze(0), torch.pow(torch.pow(lab_pred[:, 1, :, :], 2) + torch.pow(lab_pred[:, 2, :, :], 2), 0.5).unsqueeze(0))

        # edge loss
		edge_gt0, edge_gt1, edge_gt2 = fu.conv2d(y[:, 0, :, :].unsqueeze(1), mean_filter, padding=2), fu.conv2d(y[:, 1, :, :].unsqueeze(1), mean_filter, padding=2), fu.conv2d(y[:, 2, :, :].unsqueeze(1), mean_filter, padding=2)
		edge_pred0, edge_pred1, edge_pred2 = fu.conv2d(out[:, 0, :, :].unsqueeze(1), mean_filter, padding=2), fu.conv2d(out[:, 1, :, :].unsqueeze(1), mean_filter, padding=2), fu.conv2d(out[:, 2, :, :].unsqueeze(1), mean_filter, padding=2)
		edge_gt, edge_pred = (edge_gt0 + edge_gt1 + edge_gt2) / 3, (edge_pred0 + edge_pred1 + edge_pred2) / 3
		edge_loss = edeg_eval(edge_gt, edge_pred) 

		if step < 500:
		    w1, w2 = 0.1, 0.9
		else:
		    w1, w2 = 1.0, 0.001
            
		if step < 10000: 
		    for para in net.ln_conv1.parameters():
		        para.requires_grad = True
		    for para in net.ln_conv2.parameters():
		        para.requires_grad = True
		    for para in net.ln_conv3.parameters():
		        para.requires_grad = True
		    for para in net.ln_conv4.parameters():
		        para.requires_grad = True
		    for para in net.ln_conv5.parameters():
		        para.requires_grad = True
		    for para in net.ln_conv6.parameters():
		        para.requires_grad = True                
		    for para in net.ln_conv7.parameters():
		        para.requires_grad = True
		    for para in net.ln_conv8.parameters():
		        para.requires_grad = True                
		if step > 10000: 
		    for para in net.ln_conv1.parameters():
		        para.requires_grad = False
		    for para in net.ln_conv2.parameters():
		        para.requires_grad = False
		    for para in net.ln_conv3.parameters():
		        para.requires_grad = False
		    for para in net.ln_conv4.parameters():
		        para.requires_grad = False
		    for para in net.ln_conv5.parameters():
		        para.requires_grad = False
		    for para in net.ln_conv6.parameters():
		        para.requires_grad = False                
		    for para in net.ln_conv7.parameters():
		        para.requires_grad = False
		    for para in net.ln_conv8.parameters():
		        para.requires_grad = False
                
                
		loss = w1*(loss1 + edge_loss + 50*color_loss) + w2*loss2 
        
		loss.backward()
		optim.step()
		optim.zero_grad()
		losses.append(loss.item())
		print(f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}',end='',flush=True)
		torch.cuda.empty_cache()
		
		if step in itrs:
 			cc = gph
 			pic = toPIL(cc[0, :, :, :])
 			pic.save('figs/pred_grey_%d.jpg' % step)
 			cc = out
 			pic = toPIL(cc[0, :, :, :])
 			pic.save('figs/pred_%d.jpg' % step)
 			cc = y
 			pic = toPIL(cc[0, :, :, :])
 			pic.save('figs/gt_%d.jpg' % step)


		if step>=5000 and step % 1000 == 0 : 
			with torch.no_grad():
				ssim_eval, psnr_eval=test(net,loader_test, max_psnr,max_ssim,step)

			print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

			ssims.append(ssim_eval)
			psnrs.append(psnr_eval)
			if psnr_eval > max_psnr : 
				max_ssim=max(max_ssim,ssim_eval)
				max_psnr=max(max_psnr,psnr_eval)
				torch.save({
							'step':step,
							'max_psnr':max_psnr,
							'max_ssim':max_ssim,
							'ssims':ssims,
							'psnrs':psnrs,   
							'losses':losses,
							'model':net.state_dict()
				},opt.model_dir)
				print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')


def test(net,loader_test,max_psnr,max_ssim,step):
	net.eval()
	torch.cuda.empty_cache()
	ssims=[]
	psnrs=[]
	random_number = int(random.random() * 15)
	for i ,(inputs,targets) in enumerate(loader_test):
		inputs = inputs.to(opt.device); targets = targets.to(opt.device)
		pred, _ = net(inputs)
		pred = pred.unsqueeze(0)
		
		if i == random_number:            
			pic = toPIL(pred[0, :, :, :])
			pic.save('figs_val/pred.jpg')
			pic = toPIL(targets[0, :, :, :])
			pic.save('figs_val/norm.jpg')
        
		ssim1=ssim(pred, targets).item()
		psnr1=psnr(pred, targets)
		ssims.append(ssim1)
		psnrs.append(psnr1)

	return np.mean(ssims) ,np.mean(psnrs)


if __name__ == "__main__":
	loader_train=loaders_[opt.trainset]
	loader_test=loaders_[opt.testset]
	net=models_[opt.net]
	net=net.to(opt.device)
	if opt.device=='cuda':
		net=torch.nn.DataParallel(net)
		cudnn.benchmark=True
	criterion = []
	criterion.append(nn.L1Loss().to(opt.device))
	if opt.perloss:
			vgg_model = vgg16(pretrained=True).features[:16]
			vgg_model = vgg_model.to(opt.device)
			for param in vgg_model.parameters():
				param.requires_grad = False
			criterion.append(PerLoss(vgg_model).to(opt.device))
	optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()),lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
	optimizer.zero_grad()
	train(net,loader_train,loader_test,optimizer,criterion)
