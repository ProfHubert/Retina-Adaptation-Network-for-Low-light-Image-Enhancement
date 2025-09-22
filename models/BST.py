import torch.nn as nn
import torch
import torch.nn.functional as F

    
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
    
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size,):
        super(Block, self).__init__()
        self.conv1=conv(dim, dim, kernel_size, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=conv(dim,dim,kernel_size,bias=True)
        self.calayer=CALayer(dim)
        self.palayer=PALayer(dim)
    def forward(self, x):
        res=self.act1(self.conv1(x))
        res=res+x 
        res=self.conv2(res)
        res=self.calayer(res)
        res=self.palayer(res)
        res += x 
        return res
    
class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size)  for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)
    def forward(self, x):
        res = self.gp(x)
        res += x
        return res
    
class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out) # + input
        return out
    
class Dual_Tem(nn.Module):
    def __init__(self, in_ch, out_ch, scale):
        super(Dual_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1 
        )
        
        self.pa = nn.Sequential(
                nn.Conv2d(in_ch, in_ch // scale, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch // scale, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(out_ch, out_ch // scale, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch // scale, out_ch, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = out*self.pa(out) + input
        out = self.point_conv(out)
        out = out*self.ca(self.avg_pool(out))
        return out

def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """

    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules) 

class SiLU(nn.Module):  
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
class Conv_d(nn.Module):
    def __init__(self, c1, c2, dia, k=1, s=1, p=None, g=1, act=SiLU()):  
        super(Conv_d, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, 2, dilation = dia, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Substraction_term(nn.Module):
    def __init__(self, in_ch):
        super(Substraction_term, self).__init__()
        
        self.depth_conv = default_conv(in_ch, in_ch,3)
        self.point_conv = default_conv(in_ch, in_ch,1)
        self.cv2 = default_conv(in_ch, in_ch, 3)
        self.cv3 = default_conv(in_ch, in_ch, 3)
        self.cv4 = default_conv(in_ch, in_ch, 3)

    def forward(self, input): 
        f1, f2 = self.depth_conv(input), self.point_conv(input)
        diff = f1 - self.cv2(f2)
        diff = diff - self.cv4(self.cv3(diff)) 
        return diff



class BST(nn.Module):
    def __init__(self,gps,blocks,conv=default_conv):
        super(BST, self).__init__()
        
        self.gps=gps
        self.dim=64
        kernel_size=3
        self.relu = nn.ReLU(inplace=True)
        
        # HC cells
        number_f = 8
        self.ln_conv1 = conv(1,number_f,3) 
        self.ln_conv2 = conv(number_f,number_f,1) 
        self.ln_conv3 = conv(number_f,number_f,1) 
        self.ln_conv4 = conv(number_f,number_f,1)
        self.ln_conv5 = conv(number_f*2,number_f,1)
        self.ln_conv6 = conv(number_f*2,number_f,1)  
        self.ln_conv7 = conv(number_f*2,1,3) 
        self.ln_conv8 = conv(number_f*2,1,3) 
        lbias=[0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
        self.lbias = nn.Parameter(torch.FloatTensor(lbias), False)

        # BC cells
        self.dim = 64 
        vpost_precess = [conv(self.dim, self.dim, kernel_size), conv(self.dim, 3, kernel_size)]
        self.vpost = nn.Sequential(*vpost_precess)
        vpre_process = [conv(3, self.dim, kernel_size)]
        self.vpre = nn.Sequential(*vpre_process)
        self.vg1= Group(conv, self.dim, kernel_size,blocks=2) 
        self.vg2= Group(conv, self.dim, kernel_size,blocks=1) 
        self.vg3= Group(conv, self.dim, kernel_size,blocks=2) 
        self.vca=nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim*self.gps,self.dim//16,1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim//16, self.dim*self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
            ])
        self.vpalayer=PALayer(self.dim)
        self.substraction = Substraction_term(in_ch=3)
          
        
        # GC cells
        self.dim = 64
        post_precess = [conv(self.dim, self.dim, kernel_size), conv(self.dim, 3, kernel_size)]
        self.post = nn.Sequential(*post_precess)
        pre_process = [conv(3, self.dim, kernel_size)]
        self.pre = nn.Sequential(*pre_process)
        self.g1= Group(conv, self.dim, kernel_size,blocks=2) 
        self.g2= Group(conv, self.dim, kernel_size,blocks=3) 
        self.g3= Group(conv, self.dim, kernel_size,blocks=3) 
        self.ca=nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim*self.gps,self.dim//16,1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim//16, self.dim*self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
            ])
        self.palayer=PALayer(self.dim)
        
        self.cc = conv(3, 3, kernel_size)
        rod_precess = [conv(3, 3, kernel_size), conv(3, 3, kernel_size), conv(3, 3, kernel_size)]
        self.rod = nn.Sequential(*rod_precess)
        cone_precess = [conv(3, 3, kernel_size), conv(3, 3, kernel_size), conv(3, 3, kernel_size)]
        self.cone = nn.Sequential(*cone_precess)

    def forward(self, x_hsv):       

        cc = 0.25*x_hsv[:, 0,  :, :].unsqueeze(1) + 0.67*x_hsv[:, 1,  :, :].unsqueeze(1) + 0.065*x_hsv[:, 2,  :, :].unsqueeze(1)
        rod, cone = cc, x_hsv 
        
        # N-R adaption for rod
        x = rod 
        x1 = self.relu(self.ln_conv1(x))
        x2 = self.relu(self.ln_conv2(x1))
        x3 = self.relu(self.ln_conv3(x2))
        x4 = self.relu(self.ln_conv4(x3))
        x5 = self.relu(self.ln_conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.ln_conv6(torch.cat([x2,x5],1)))
        x_n = F.tanh(self.ln_conv7(torch.cat([x1,x6],1)))
        x_s = F.tanh(self.ln_conv8(torch.cat([x1,x6],1)))
        
        x_n = nn.Sigmoid()(x_n)
        cc = x_s.max()
        if x_s.max() < 20: 
            cc = 20
        else:
            cc = cc.item()
        x_s = x_s.clamp(0, cc)
        
        if torch.isnan(x_s).any():
            x_s = torch.where(torch.isnan(x_s), torch.full_like(x_s, 0), x_s)
        if torch.isnan(x_n).any():
            x_n = torch.where(torch.isnan(x_n), torch.full_like(x_n, 0), x_n)
        
        enhanced_rod = torch.pow(x_hsv, x_n) / ((torch.pow(x_hsv, x_n) + torch.pow(x_s, x_n)) + self.lbias[0])                 
        
        # enhancement for BC cells
        x = cone 
        lf1 = self.vpre(x)
        lres1=self.vg1(lf1)
        lres2=self.vg2(lres1)
        lres3=self.vg3(lres2)
        lw=self.vca(torch.cat([lres1,lres2,lres3],dim=1))
        lw=lw.view(-1,self.gps,self.dim)[:,:,:,None,None]
        lout=lw[:,0,::]*lres1+lw[:,1,::]*lres2+lw[:,2,::]*lres3
        lout=self.vpalayer(lout)
        enhanced_cone = self.vpost(lout) + x_hsv  
        
        x_fusion = self.cone(enhanced_cone) + 0.1*self.rod(enhanced_rod) 
        x_fusion = x_fusion + x_hsv
        x_fusion = self.substraction(x_fusion) 
        x_fusion = self.cc(x_fusion) + x_hsv

        # enhancement for GC cells
        hf1 = self.pre(x_fusion)
        res1=self.g1(hf1)
        res2=self.g2(res1)
        res3=self.g3(res2)
        w=self.ca(torch.cat([res1,res2,res3],dim=1))
        w=w.view(-1,self.gps,self.dim)[:,:,:,None,None]
        out=w[:,0,::]*res1+w[:,1,::]*res2+w[:,2,::]*res3
        out=self.palayer(out)
        fu = self.post(out) + x_fusion 
        fu = nn.Sigmoid()(fu)
        fu = torch.squeeze(fu.clamp(0,1))
    
        return fu, enhanced_rod.clamp(0,1) 


if __name__ == '__main__':
    
    import numpy as np
    x = torch.tensor(np.random.rand(2,3,480,640).astype(np.float32))
    model = BST(gps=3, blocks=3)
    y = model(x)[0]
    print('output shape:', y.shape)
    print('test ok!')
    torch.save(model, "cc.pth")