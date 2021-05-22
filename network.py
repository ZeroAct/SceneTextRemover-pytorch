import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import \
    _double_conv2d, _down_conv2d, _up_conv2d, _final_conv2d, _dis_conv, _one_conv

from losses import TSDLoss, TRGLoss

# Text Stroke Detection (GD in paper)
class TSDNet(nn.Module):
    def __init__(self):
        super(TSDNet, self).__init__()

        self.inc = _double_conv2d(4, 16, 3)
        self.down1 = _down_conv2d(16, 32, 3)
        self.down2 = _down_conv2d(32, 64, 3)
        self.down3 = _down_conv2d(64, 128, 3)
        
        self.up1 = _up_conv2d(128, 64, 3)
        self.up2 = _up_conv2d(64, 32, 3)
        self.up3 = _up_conv2d(32, 16, 3)
        
        self.outc = _final_conv2d(16, 1, 3)

    def forward(self, Igt, M):
        x = torch.cat([Igt, M], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        M = self.outc(x)
        return M

# Text Removal Generation (GR, GR' in paper)
class TRGNet(nn.Module):
    def __init__(self):
        super(TRGNet, self).__init__()

        self.inc = _double_conv2d(5, 16, 5, 2)
        self.down1 = _down_conv2d(16, 32, 3)
        self.down2 = _down_conv2d(32, 64, 3)
        self.down3 = _down_conv2d(64, 128, 3)
        
        self.mid_layer = _double_conv2d(128, 128, 3)
        
        self.up1 = _up_conv2d(128, 64, 3)
        self.up2 = _up_conv2d(64, 32, 3)
        self.up3 = _up_conv2d(32, 16, 3)
        
        self.outc = _final_conv2d(16, 3, 3)

    def forward(self, Igt, M, Ms):
        x = torch.cat([Igt, M, Ms], dim=1)
        x1 = self.inc(x)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x4 = torch.add(self.mid_layer(x4), x4)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        M = self.outc(x)
        return M

# Text Stroke Detection _ (G'D in paper)
class TSDNet_(nn.Module):
    def __init__(self):
        super(TSDNet_, self).__init__()

        self.inc = _double_conv2d(5, 16, 3)
        self.down1 = _down_conv2d(16, 32, 3)
        self.down2 = _down_conv2d(32, 64, 3)
        self.down3 = _down_conv2d(64, 128, 3)
        
        self.up1 = _up_conv2d(128, 64, 3)
        self.up2 = _up_conv2d(64, 32, 3)
        self.up3 = _up_conv2d(32, 16, 3)
        
        self.outc = _final_conv2d(16, 1, 3)

    def forward(self, Ite, M, Ms):
        x = torch.cat([Ite, M, Ms], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        M = self.outc(x)
        return M

# weighted patch based discriminator (D, Dm in paper)
# build_sn_patch_gan_discriminator 
# (https://github.com/JiahuiYu/generative_inpainting/blob/master/inpaint_model.py)
class Discriminator(nn.Module):
    def __init__(self):
        
        super(Discriminator, self).__init__()
        
        self.Dm = nn.Sequential(
                _one_conv(1, 1, 5, 2, 2),
                nn.Sigmoid(),
                _one_conv(1, 1, 5, 2, 2),
                nn.Sigmoid(),
                _one_conv(1, 1, 5, 2, 2),
                nn.Sigmoid(),
                _one_conv(1, 1, 5, 2, 2),
                nn.Sigmoid(),
                _one_conv(1, 1, 5, 2, 2),
                nn.Sigmoid()
            )
        
        self.D = nn.Sequential(
                _dis_conv(3, 64, 5, 2, 2),
                _dis_conv(64, 128, 5, 2, 2),
                _dis_conv(128, 256, 5, 2, 2),
                _dis_conv(256, 256, 5, 2, 2),
                _dis_conv(256, 256, 5, 2, 2)
            )
        
        self.pool = nn.AvgPool2d(8)
        self.linear = nn.Linear(256, 1)
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, Mm, Ite_):
        mi = self.Dm(Mm)
        di = self.D(Ite_)
        
        y = torch.mul(mi, di)
        # y = self.pool(y)
        # y = self.linear(y.view(-1, 256))
        return y


class STRNet(nn.Module):
    def __init__(self):
        
        super(STRNet, self).__init__()
        
        self.tsdnet  = TSDNet()
        self.trgnet  = TRGNet()
        self.tsdnet_ = TSDNet_()
        self.trgnet_ = TRGNet()
        
        self.discrim = Discriminator()
    
    def forward(self, I, Mm):
        Ms = self.tsdnet(I, Mm)
        Ite = self.trgnet(I, Mm, Ms)
        Ms_ = self.tsdnet_(Ite, Mm, Ms)
        Ite_ = self.trgnet_(Ite, Mm, Ms)
        
        return Ms, Ite, Ms_, Ite_
    

if __name__ == "__main__":
    from torch.optim import Adam
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    print(device)
    
    # I    : input image
    # Itegt: input image
    # M    : Text Region mask
    # Ms   : Text Stroke mask (from tsdnet)
    # 
    
    I = torch.randn((2, 3, 256, 256)).to(device)
    print(f"I shape\n : {I.shape}")
    
    Itegt = torch.randn((2, 3, 256, 256)).to(device)
    print(f"Itegt shape\n : {Itegt.shape}")
    
    Mm = torch.randn((2, 1, 256, 256)).to(device)
    print(f"Mm shape\n : {Mm.shape}")
    
    Msgt = torch.randn((2, 1, 256, 256)).to(device)
    print(f"Mgt shape\n : {Msgt.shape}")
    
    One = torch.ones((2, 1)).to(device)
    Zero = torch.zeros((2, 1)).to(device)
    
    model = STRNet().to(device)
    
    model_optim = Adam(model.parameters(), 0.0001)
    discrim_optim = Adam(model.discrim.parameters(), 0.0001)
    bce_loss = nn.BCEWithLogitsLoss()
    
    Ms, Ite, Ms_, Ite_ = model.forward(I, Mm)
    
    Ltsd = TSDLoss(Msgt, Ms, Ms_)
    Ltrg = TRGLoss(Mm, Ms, Ms_, Itegt, Ite, Ite_)
    # Lgsn = -bce_loss(model.discrim(Mm, Ite_), One)
    Lgsn = -torch.mean(model.discrim(Mm, Ite_))
    
    total_loss = Ltsd + Ltrg + Lgsn
    
    model_optim.zero_grad()
    total_loss.backward()
    model_optim.step()
    
    Ms, Ite, Ms_, Ite_ = model.forward(I, Mm)
    # Ldsn = F.relu(1-bce_loss(model.discrim(Mm, Itegt), One)) + \
    #               F.relu(1+bce_loss(model.discrim(Mm, Ite_), Zero))
    Ldsn = torch.mean(F.relu(1-model.discrim(Mm, Itegt))) + \
                  torch.mean(F.relu(1+model.discrim(Mm, Ite_)))
                  
    discrim_optim.zero_grad()
    Ldsn.backward()
    discrim_optim.step()
    
    
    # Igt = torch.randn((2, 3, 256, 256)).to(device)
    # print(f"Igt shape\n : {Igt.shape}")
    
    # M = torch.randn((2, 1, 256, 256)).to(device)
    # print(f"M shape\n : {M.shape}")
    
    # Mgt = torch.randn((2, 1, 256, 256)).to(device)
    # print(f"Mgt shape\n : {Mgt.shape}")
    
    # # models
    # tsdnet = TSDNet().to(device)
    # trgnet = TRGNet().to(device)
    # tsdnet_ = TSDNet_().to(device)
    # trgnet_ = TRGNet().to(device)
    
    # discriminator = Discriminator().to(device)
    
    # # optim
    # from torch.optim import Adam
    # total_optim = Adam(list(tsdnet.parameters()) + list(trgnet.parameters()) +
    #                     list(tsdnet_.parameters()) + list(trgnet_.parameters()), 0.0001)
    # total_optim.zero_grad()
    
    # discr_optim = Adam(discriminator.parameters())
    # discr_optim.zero_grad()
    
    # # inference
    # Ms = tsdnet(Igt, M)
    # # print(f"tsdnet output Ms shape\n : {Ms.shape}")
    # Ite = trgnet(Igt, M, Ms)
    # # print(f"trgnet output Ite shape\n : {Ite.shape}")
    # Ms_ = tsdnet_(Ite, M, Ms)
    # # print(f"tsdnet_ output Ms_ shape\n : {Ms_.shape}")
    # Ite_ = trgnet_(Ite, M, Ms_)
    # # print(f"Final trgnet_ output Ite_ shape\n : {Ite.shape}")
    
    # # calculate loss
    # Lgsn = -discriminator.forward_with_loss(M, Ite)
    
    # from losses import TSDLoss, TRGLoss
    # Ltsd = TSDLoss(Mgt, Ms, Ms_)
    # Ltrg = TRGLoss(M, Ms, Ms_, Igt, Ite, Ite_)
    
    # total_loss = Ltsd + Ltrg + Lgsn
    
    # # train total model 
    # total_loss.backward()
    # total_optim.step()
    # print(total_loss.detach().cpu().item())
    
    # # train discriminator
    # Ms = tsdnet(Igt, M)
    # # print(f"tsdnet output Ms shape\n : {Ms.shape}")
    # Ite = trgnet(Igt, M, Ms)
    # # print(f"trgnet output Ite shape\n : {Ite.shape}")
    # Ms_ = tsdnet_(Ite, M, Ms)
    # # print(f"tsdnet_ output Ms_ shape\n : {Ms_.shape}")
    # Ite_ = trgnet_(Ite, M, Ms_)
    # # print(f"Final trgnet_ output Ite_ shape\n : {Ite.shape}")
    
    # Ldsn = discriminator.forward_with_loss(M, Igt) + discriminator.forward_with_loss(M, Ite_)
    # discr_loss = Ldsn
    
    # discr_loss.backward()
    # discr_optim.step()
    # print(discr_loss.detach().cpu().item())
