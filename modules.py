import torch
import torch.nn as nn
import torch.nn.functional as F

# dis_conv 
# (https://github.com/JiahuiYu/generative_inpainting/blob/3a5324373ba52c68c79587ca183bc10b9e57b783/inpaint_ops.py#L84)
class _dis_conv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2):
        super().__init__()
        
        self._conv = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            ),
            nn.LeakyReLU(inplace=True)
        )
        
        # weight initialization
        def weight_init(m):
            if isinstance(m, nn.Conv2d):
                # nn.utils.spectral_norm(m.weight)
                nn.init.zeros_(m.bias)
        
        self.apply(weight_init)

    def forward(self, x):
        return self._conv(x)

# weights are fixed to one, bias to zero
class _one_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2):
        super().__init__()
        
        self._conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )
        
        # weight initialization
        def weight_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                m.weight.requires_grad = False
                m.bias.requires_grad = False
        
        self.apply(weight_init)

    def forward(self, x):
        return self._conv(x)

class _double_conv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, mid_channels=None):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # weight initialization
        def weight_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
        
        self.apply(weight_init)

    def forward(self, x):
        return self.double_conv(x)


class _down_conv2d(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        
        super().__init__()
        
        self.seq_model = nn.Sequential(
                nn.MaxPool2d(2),
                _double_conv2d(in_channels, out_channels)
            )
        
        
    def forward(self, x):
        return self.seq_model(x)


class _up_conv2d(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        
        super().__init__()
        
        self.conv_t = nn.ConvTranspose2d(in_channels, in_channels//2, 2, 2)
        self.conv   = _double_conv2d(in_channels, out_channels)
        
    # x1 : input, x2 : matching down_conv2d output
    def forward(self, x1, x2):
        x1 = self.conv_t(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class _final_conv2d(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1)
        
    def forward(self, x):
        return self.conv(x)