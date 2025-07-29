import torch
import torch.nn as nn
from collections import OrderedDict



class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 3, 3), dilation=(1, 1, 1), bias=False):
        super().__init__()
        
        time_pad = (kernel_size[0] - 1) * dilation[0]
        height_pad = ((kernel_size[1] - 1) * dilation[1]) // 2
        width_pad = ((kernel_size[2] - 1) * dilation[2]) // 2

        # Pad temporally on the left
        # padding (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
        self.pad = nn.ConstantPad3d(padding=(width_pad, width_pad, height_pad, height_pad, time_pad, 0), value=0)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, dilation=dilation, stride=1, padding=0, bias=bias)
        self.norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, *inputs):
        (x,) = inputs
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class TemporalConv3d(nn.Module):  # meiyong
    def __init__(self, in_channels, out_channels, n_present, n_future, kernel_size=(2, 3, 3), dilation=(1, 1, 1), bias=False):
        super(TemporalConv3d, self).__init__()
        
        time_pad = n_future - n_present + dilation[0] * (kernel_size[0] - 1)
        height_pad = ((kernel_size[1] - 1) * dilation[1]) // 2
        width_pad = ((kernel_size[2] - 1) * dilation[2]) // 2

        self.pad = nn.ConstantPad3d(padding=(width_pad, width_pad, height_pad, height_pad, time_pad // 2, time_pad - time_pad // 2), value=0)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, dilation=dilation, stride=1, padding=0, bias=bias)
        self.norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class CausalMaxPool3d(nn.Module): # meiyong
    def __init__(self, kernel_size=(2, 3, 3)):
        super().__init__()
        
        time_pad = kernel_size[0] - 1
        height_pad = (kernel_size[1] - 1) // 2
        width_pad = (kernel_size[2] - 1) // 2

        # Pad temporally on the left
        self.pad = nn.ConstantPad3d(padding=(width_pad, width_pad, height_pad, height_pad, time_pad, 0), value=0)
        self.max_pool = nn.MaxPool3d(kernel_size, stride=1)

    def forward(self, *inputs):
        (x,) = inputs
        x = self.pad(x)
        x = self.max_pool(x)
        return x


def conv_1x1x1_norm_activated(in_channels, out_channels):
    return nn.Sequential(
        OrderedDict(
            [
                ('conv', nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)),
                ('norm', nn.BatchNorm3d(out_channels)),
                ('activation', nn.ReLU(inplace=True)),
            ]
        )
    )


class Bottleneck3D(nn.Module):

    def __init__(self, in_channels, out_channels=None, kernel_size=(2, 3, 3), dilation=(1, 1, 1)):
        super().__init__()
        bottleneck_channels = in_channels // 2
        out_channels = out_channels or in_channels

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    #
                    ('conv_down_project', conv_1x1x1_norm_activated(in_channels, bottleneck_channels)),
                    # Second conv block
                    (
                        'conv',
                        CausalConv3d(
                            bottleneck_channels,
                            bottleneck_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            bias=False,
                        ),
                    ),
                    # Final projection with 1x1 kernel
                    ('conv_up_project', conv_1x1x1_norm_activated(bottleneck_channels, out_channels)),
                ]
            )
        )

        if out_channels != in_channels:
            self.projection = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.projection = None

    def forward(self, *args):
        (x,) = args
        x_residual = self.layers(x)
        x_features = self.projection(x) if self.projection is not None else x
        return x_residual + x_features



# class TemporalBlock(nn.Module):
#     def __init__(self, in_channels, out_channels=None):
#         super().__init__()
#         self.in_channels = in_channels
#         self.half_channels = in_channels // 2
#         self.out_channels = out_channels or self.in_channels
#         self.kernels = [(2, 3, 3), (1, 3, 3)]
#         self.convolution_paths = []
#         for kernel_size in self.kernels:
#             self.convolution_paths.append(
#                 nn.Sequential(
#                     conv_1x1x1_norm_activated(self.in_channels, self.half_channels),
#                     CausalConv3d(self.half_channels, self.half_channels, kernel_size=kernel_size),
#                 )
#             )
#         self.convolution_paths.append(conv_1x1x1_norm_activated(self.in_channels, self.half_channels))
#         self.convolution_paths = nn.ModuleList(self.convolution_paths)

#         agg_in_channels = len(self.convolution_paths) * self.half_channels
#         self.aggregation = nn.Sequential(
#             conv_1x1x1_norm_activated(agg_in_channels, self.out_channels),)

#         if self.out_channels != self.in_channels:
#             self.projection = nn.Sequential(
#                 nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
#                 nn.BatchNorm3d(self.out_channels),
#             )
#         else:
#             self.projection = None

#     def forward(self, *inputs):
#         (x,) = inputs
#         x_paths = []
#         for conv in self.convolution_paths:
#             x_paths.append(conv(x))
#         x_residual = torch.cat(x_paths, dim=1)
#         x_residual = self.aggregation(x_residual)

#         if self.out_channels != self.in_channels:
#             x = self.projection(x)
#         x = x + x_residual
#         return x




class TemporalBlock(nn.Module):


    def __init__(self, in_channels, out_channels=None, use_pyramid_pooling=False, pool_sizes=None):
        super().__init__()
        self.in_channels = in_channels
        self.half_channels = in_channels // 2
        self.out_channels = out_channels or self.in_channels
        self.kernels = [(2, 3, 3), (1, 3, 3)]

        #  
        self.use_pyramid_pooling = use_pyramid_pooling

        #             
        self.convolution_paths = []
        for kernel_size in self.kernels:
            self.convolution_paths.append(
                nn.Sequential(
                    conv_1x1x1_norm_activated(self.in_channels, self.half_channels),
                    CausalConv3d(self.half_channels, self.half_channels, kernel_size=kernel_size),
                )
            )
        self.convolution_paths.append(conv_1x1x1_norm_activated(self.in_channels, self.half_channels))
        self.convolution_paths = nn.ModuleList(self.convolution_paths)

        agg_in_channels = len(self.convolution_paths) * self.half_channels

        # 
        self.aggregation = nn.Sequential(
            conv_1x1x1_norm_activated(agg_in_channels, self.out_channels),)

        if self.out_channels != self.in_channels:
            self.projection = nn.Sequential(
                nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(self.out_channels),
            )
        else:
            self.projection = None

    def forward(self, *inputs):
        (x,) = inputs
        x_paths = []
        for conv in self.convolution_paths:
            x_paths.append(conv(x))
        x_residual = torch.cat(x_paths, dim=1)
        #  
        x_residual = self.aggregation(x_residual)

        if self.out_channels != self.in_channels:
            x = self.projection(x)
        x = x + x_residual
        return x

