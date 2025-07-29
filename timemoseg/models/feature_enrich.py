import torch
import torch.nn as nn

from timemoseg.layers.temporal import Bottleneck3D, TemporalBlock
from timemoseg.layers.convolutions import ConvBlock, Bottleneck, DeepLabHead, TripletAttention

# from timemoseg.models.feature_enrich import  EnrichModel, Rich3d



class MultiConv(nn. Module):
    def __init__(self, inp, oup):
        super(MultiConv, self).__init__()

        self.groups = oup // 4
        in_channel = inp // 4
        out_channel = oup // 4

        self.dwconv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.dwconv2 = nn.Conv2d(in_channel, out_channel, 5, padding=2)
        self.dwconv3 = nn.Conv2d(in_channel, out_channel, 7, padding=3)
        self.dwconv4 = nn.Conv2d(in_channel, out_channel, 9, padding=4)

        self.finalconv = nn.Sequential(
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
            nn.Conv2d(oup, oup, 1),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print(x.shape)
        a, b, c, d = torch.split(x, self.groups, dim=1)
        a = self.dwconv1(a)
        b = self.dwconv1(b)
        c = self.dwconv1(c)
        d = self.dwconv1(d)

        out = torch.cat([a, b, c, d], dim=1)
        out = self.finalconv(out)

        return  out

class TemporalModel(nn.Module):
    def __init__(
            self, in_channels, receptive_field,  start_out_channels=128, extra_in_channels=0,
            n_spatial_layers_between_temporal_layers=0, use_pyramid_pooling=True):
        super().__init__()
        self.receptive_field = receptive_field
        n_temporal_layers = receptive_field - 1  # 2

        h, w = 200 ,200
        modules = []

        block_in_channels = in_channels  # 64 
        block_out_channels = start_out_channels  # 64

        for _ in range(n_temporal_layers):
            if use_pyramid_pooling:
                use_pyramid_pooling = True
                pool_sizes = [(2, h, w)]
            else:
                use_pyramid_pooling = False
                pool_sizes = None
            temporal = TemporalBlock(
                block_in_channels,
                block_out_channels,
                use_pyramid_pooling=use_pyramid_pooling,
                pool_sizes=pool_sizes,
            )
            spatial = [
                Bottleneck3D(block_out_channels, block_out_channels, kernel_size=(1, 3, 3))
                for _ in range(n_spatial_layers_between_temporal_layers)
            ]
            temporal_spatial_layers = nn.Sequential(temporal, *spatial)
            modules.extend(temporal_spatial_layers)

            block_in_channels = block_out_channels
            block_out_channels += extra_in_channels

        self.out_channels = block_in_channels

        
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        # 
        x = x.permute(0, 2, 1, 3, 4) # t 
        x = self.model(x) #  
        x = x.permute(0, 2, 1, 3, 4).contiguous()  
        
        return x




class CAM3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel))
        self.spatial_se = nn.Conv3d(channel, 1, kernel_size=1,
                                    stride=1, padding=0, bias=False)

    def forward(self, x):
        bahs, chs, _, _, _ = x.size()
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = torch.sigmoid(self.channel_excitation(chn_se).view(bahs, chs, 1, 1,1))
        chn_se = torch.mul(x, chn_se)
        spa_se = torch.sigmoid(self.spatial_se(x))
        spa_se = torch.mul(x, spa_se)
        net_out = spa_se + x + chn_se
        return net_out

class Rich3d(nn.Module):
    def __init__(self, in_channels  ):
        super().__init__()
        self.cam3d= CAM3D(in_channels )
        self.out_channels = in_channels
        self.final_conv = DeepLabHead(self.out_channels, self.out_channels, hidden_channel=128)  # official 
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4) # torch.Size([2, 70, 3, 200, 200])
        x = self.cam3d(x) # torch.Size([2, 64, 3, 200, 200])
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # time 又换回来了[2, 3, 64, 200, 200])

        b, s, c, h, w = x.shape # 2  3 64 200 200 
        x = x.view(b * s, c, h, w)
        # x = self.rich_conv(x)
        x = self.final_conv(x) # 加点conv 特征丰富一哈([6, 64, 200, 200])
        x = x.view(b, s, c, h, w) #  2 3 64 200 200
        return x