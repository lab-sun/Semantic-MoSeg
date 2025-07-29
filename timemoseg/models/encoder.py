import torch.nn as nn
import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from timemoseg.layers.convolutions import UpsamplingConcat, DeepLabHead, Depthrich
from torchvision.models.resnet import Bottleneck

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


class DepthAttention(nn.Module):

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        # self.dim=dim
        self.kernel_size = kernel_size

        self.value = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU())
        self.key_embed = ResNetBottleNeck(dim)
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // 4, 1, bias=False),
            nn.BatchNorm2d(2 * dim // 4),
            nn.ReLU(),
            nn.Conv2d(2 * dim // 4, kernel_size * kernel_size * dim, 1)
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)
        v = self.value(x).view(bs, c, -1)
        y = torch.cat([k1, x], dim=1)
        att = self.attention_embed(y)
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)
        feature = F.softmax(att, dim=-1) * v
        feature =feature.view(bs, c, h, w)

        return k1 + feature


class MultiConv(nn.Module):
    def __init__(self, inp, oup):
        super(MultiConv, self).__init__()

        self.groups = oup // 3
        in_channel = inp // 3
        out_channel = oup // 3

        self.dwconv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.dwconv2 = nn.Conv2d(in_channel, out_channel, 5, padding=2)
        self.dwconv3 = nn.Conv2d(in_channel, out_channel, 7, padding=3)
        # self.dwconv4 = nn.Conv2d(in_channel, out_channel, 9, padding=4)

        self.finalconv = nn.Sequential(
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
            nn.Conv2d(oup, oup, 1),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        a, b, c = torch.split(x, self.groups, dim=1)
        a = self.dwconv1(a)
        b = self.dwconv1(b)
        c = self.dwconv1(c)
        # d = self.dwconv1(d)

        out = torch.cat([a, b, c], dim=1)
        out = self.finalconv(out)

        return out


class Encoder(nn.Module):
    def __init__(self, cfg, D):
        super().__init__()
        self.D = D  # 48
        self.C = cfg.OUT_CHANNELS
        self.downsample = cfg.DOWNFACTOR
        self.version = cfg.NAME.split('-')[1]
        self.backbone = EfficientNet.from_name(cfg.NAME)
        self.delete_unused_layers()
        if self.version == 'b4':
            self.reduction_channel = [0, 24, 32, 56, 160, 448]

        self.backbone._conv_stem = nn.Conv2d(6, 48, kernel_size=3, stride=2, padding=1, bias=False)

        index = np.log2(self.downsample).astype(np.int)
        self.depth_layer_1 = DeepLabHead(self.reduction_channel[index + 1], self.reduction_channel[index + 1],
                                         hidden_channel=64)
        self.depth_layer_2 = UpsamplingConcat(self.reduction_channel[index + 1] + self.reduction_channel[index], self.D)

        self.feature_layer_1 = DeepLabHead(self.reduction_channel[index + 1], self.reduction_channel[index + 1],
                                           hidden_channel=64)
        self.feature_layer_2 = UpsamplingConcat(self.reduction_channel[index + 1] + self.reduction_channel[index],
                                                self.C)
        self.mlp = nn.Sequential(
            nn.BatchNorm2d(21),
            nn.Conv2d(21, self.D, 1),
            nn.ReLU(),
        )
        self.propagate = ResNetBottleNeck(self.D)
        self.depthatten = Depthrich(self.D, self.D)


    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features_depth(self, x):
        endpoints = dict()
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b4' and idx == 21:
                    break

        
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        index = np.log2(self.downsample).astype(np.int)
        input_1 = endpoints['reduction_{}'.format(index + 1)]
        input_2 = endpoints['reduction_{}'.format(index)]

        feature = self.feature_layer_1(input_1)
        feature = self.feature_layer_2(feature, input_2)

        # 
        depth = self.depth_layer_1(input_1)
        depth = self.depth_layer_2(depth, input_2)

        return feature, depth

    def forward(self, x, intrinsics, translation, rotation):
        intrinsics = intrinsics.unsqueeze(2).unsqueeze(3)
        translation = translation.unsqueeze(2).unsqueeze(3)
        rotation = rotation.unsqueeze(2).unsqueeze(3)
        feature, depth = self.get_features_depth(x)
        geometric_param = torch.cat([intrinsics, translation, rotation], dim=1)
        geometric_param = self.mlp(geometric_param)

        depth = depth * F.sigmoid(geometric_param)
        depth = self.propagate(depth)
        depth = self.depthatten(depth)
        return feature, depth
