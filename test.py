import os,argparse
import numpy as np
from PIL import Image
from metrics import psnr,ssim
from models import *
from data_utils import *
from rgb2lab import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs 
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import transforms
import lpips
toPIL = transforms.ToPILImage()
import time

abs=os.getcwd()+'/'
def tensorShow(tensors,titles=['haze']):
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(221+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

parser=argparse.ArgumentParser()
parser.add_argument('--task',type=str,default='its',help='its or ots')
parser.add_argument('--test_imgs',type=str,default='test_imgs',help='Test imgs folder')
opt=parser.parse_args()
dataset=opt.task
gps=3
blocks=3
img_dir= "LOL real data\\test\\low\\"
gt_dir = "LOL real data\\test\\high\\"


output_dir=abs+f'pred_FFA_{dataset}/'
print("pred_dir:",output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_dir=abs+f'trained_models/{dataset}_train_bst_{gps}_{blocks}.pk'
device='cuda' if torch.cuda.is_available() else 'cpu'
ckp=torch.load(model_dir,map_location=device,weights_only=False)
net=BST(gps=gps,blocks=blocks).cuda() 
net=nn.DataParallel(net)
net = net.module
net.load_state_dict(ckp['model'])
net.eval()
convertor = RGB_HSV()
ssims=[]
psnrs=[]
times = []
lpipses=[]

loss_fn = lpips.LPIPS(net='alex', spatial = True).cuda()

for im in os.listdir(img_dir):
    print(f'\r {im}',end='',flush=True)
    lows  = Image.open(img_dir+im)
    highs = Image.open(gt_dir+im)
    
    low= tfs.Compose([
        tfs.ToTensor()
    ])(lows)[None,::] 
    high= tfs.Compose([
        tfs.ToTensor()
    ])(highs)[None,::]

    
    with torch.no_grad():
        start = time.time()
        pred, z = net(low.cuda())
        end_time = (time.time() - start)
        times.append(end_time)
        pred = pred.unsqueeze(0)

        ssim1=ssim(pred, high.cuda()).item()
        psnr1=psnr(pred, high.cuda())
        ex_d1 = loss_fn.forward(high.cuda(), pred).mean()
        
        ssims.append(ssim1)
        psnrs.append(psnr1)   
        lpipses.append(ex_d1.item())      

    pic = toPIL(pred[0, :, :, :])
    pic.save(output_dir+im.split('.')[0]+'.png')
    
score_ssim, score_psnr, score_lpips = np.mean(ssims), np.mean(psnrs), np.mean(lpipses)
print("\nscore psnr & ssim & lpips", score_psnr, ", ", score_ssim, ", ", score_lpips)
print('\n running time for each image:', np.mean(times))
