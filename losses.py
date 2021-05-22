import torch
import torch.nn as nn
import torch.nn.functional as F

def TSDLoss(Mgt, Ms, Ms_, r=10):
    return torch.mean(torch.abs(Ms-Mgt) + r * torch.abs(Ms_-Mgt))

def TRGLoss(Mm, Ms, Ms_, Itegt, Ite, Ite_, rm=5, rs=5, rr=10):
    
    Mw  = torch.ones_like(Mm) + rm * Mm + rs * Ms
    Mw_ = torch.ones_like(Mm) + rm * Mm + rs * Ms_
    
    Ltrg = torch.mean(torch.abs(torch.mul(Ite, Mw) - torch.mul(Itegt, Mw)) + \
                     rr * torch.abs(torch.mul(Ite_, Mw_) - torch.mul(Itegt, Mw_)))
    
    return Ltrg
