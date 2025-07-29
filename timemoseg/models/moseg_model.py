import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import Bottleneck
from timemoseg.models.encoder import Encoder
from timemoseg.models.feature_enrich import  TemporalModel,Rich3d
#
from timemoseg.models.decoder_densenet import Decoder
from timemoseg.utils.network import pack_sequence_dim, unpack_sequence_dim, set_bn_momentum
from timemoseg.utils.geometry import calculate_bev_params, VoxelsSumming, pose_vec2mat

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)  # 3 h w
    indices = indices[None]  #

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    sh = h / h_meters
    sw = w / w_meters

    return [
        [0., -sw, w / 2.],
        [-sh, 0., h * offset + h / 2.],
        [0., 0., 1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class MOSEG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        bev_resolution, bev_start_position, bev_dimension = calculate_bev_params( self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND
        )
        self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        self.bev_start_position = nn.Parameter(bev_start_position, requires_grad=False)
        self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)  # 200, 200

        self.encoder_downfactor = self.cfg.MODEL.ENCODER.DOWNFACTOR  # 8
        self.encoder_out_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS

        self.frustum = self.create_frustum()
        self.depth_channels, _, _, _ = self.frustum.shape
        self.receptive_field = self.cfg.TIME_RECEPTIVE_FIELD

        #
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])
        self.bev_size = (self.bev_dimension[0].item(), self.bev_dimension[1].item())
        self.encoder = Encoder(cfg=self.cfg.MODEL.ENCODER, D=self.depth_channels)
        self.temporal_model = TemporalModel(
            self.encoder_out_channels,
            self.receptive_field,
            )
        # Decoder
        self.decoder = Decoder(
            in_channels=self.encoder_out_channels,
            n_classes=2,
            bevfeature={
                'predict_segmentation': self.cfg.SEMANTIC_SEG.VEHICLE.ENABLED,
                'moving_seg': self.cfg.MOVING_SEG.ENABLED,
            } )

        set_bn_momentum(self, self.cfg.MODEL.BN_MOMENTUM)

    def create_frustum(self):
        h, w = self.cfg.IMAGE.FINAL_DIM
        downsampled_h, downsampled_w = h // self.encoder_downfactor, w // self.encoder_downfactor

        depth_grid = torch.arange(*self.cfg.LIFT.D_BOUND, dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, downsampled_h, downsampled_w)
        n_depth_slices = depth_grid.shape[0]

        x_grid = torch.linspace(0, w - 1, downsampled_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, downsampled_w).expand(n_depth_slices, downsampled_h, downsampled_w)
        y_grid = torch.linspace(0, h - 1, downsampled_h, dtype=torch.float)
        y_grid = y_grid.view(1, downsampled_h, 1).expand(n_depth_slices, downsampled_h, downsampled_w)

        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def featur2d_compute(self, x, intrinsics, extrinsics, bt_egomotion):
        b, n, c, h, w = x.shape  #

        x = x.view(b * n, c, h, w)
        intrinsics = intrinsics.view(b * n, -1)
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        rotation = rotation.reshape(b * n, -1)
        translation = translation.reshape(b * n, -1)
        x, depth = self.encoder(x, intrinsics, translation, rotation)
        depth_prob = depth.softmax(dim=1)  #
        x = depth_prob.unsqueeze(1) * x.unsqueeze(2)
        x = x.view(b, n, *x.shape[1:])  #
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x, depth_prob

    def get_geometry(self, intrinsics, extrinsics):

        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        B, N, _ = translation.shape
        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combined_transformation = rotation.matmul(torch.inverse(intrinsics))
        points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += translation.view(B, N, 1, 1, 1, 3)

        return points

    def bev_features(self, x, intrinsics, extrinsics, vehicle_egomotion):
        b, s, n, c, h, w = x.shape
        x = pack_sequence_dim(x)
        intrinsics = pack_sequence_dim(intrinsics)
        extrinsics = pack_sequence_dim(extrinsics)
        bt_egomotion = vehicle_egomotion.unsqueeze(2)
        bt_egomotion = torch.cat([bt_egomotion, bt_egomotion, bt_egomotion, bt_egomotion, bt_egomotion, bt_egomotion],
                                 dim=2)
        bt_egomotion = pack_sequence_dim(bt_egomotion)
        geometry = self.get_geometry(intrinsics, extrinsics)
        x, depth_prob = self.featur2d_compute(x, intrinsics, extrinsics, bt_egomotion)
        x = unpack_sequence_dim(x, b, s)
        geometry = unpack_sequence_dim(geometry, b, s)

        x = self.bev_generate_fea(x, geometry, vehicle_egomotion)
        return x, depth_prob

    def forward(self, image, intrinsics, extrinsics, vehicle_egomotion):
        output = {}
        image = image[:, :self.receptive_field].contiguous()
        b, s, n, c, _, _ = image.shape
        intrinsics = intrinsics[:, :self.receptive_field].contiguous()
        extrinsics = extrinsics[:, :self.receptive_field].contiguous()
        vehicle_egomotion = vehicle_egomotion[:, :self.receptive_field].contiguous()
        x, depth  = self.bev_features(image, intrinsics, extrinsics, vehicle_egomotion)  #

        output = {**output, 'depth_prob': depth, }
        states = self.temporal_model(x)
        bev_output = self.decoder(states)
        output = {**output, **bev_output}

        return output


    def bev_generate_fea(self, x, geometry, vehicle_egomotion):

        batch, s, n, d, h, w, c = x.shape
        output = torch.zeros(
            (batch, s, c, self.bev_dimension[0], self.bev_dimension[1]), dtype=torch.float, device=x.device
        )

        egomotion_mat = pose_vec2mat(vehicle_egomotion)
        rotation, translation = egomotion_mat[..., :3, :3], egomotion_mat[..., :3, 3]

        def voxel_to_pixel(geometry_b, x_b):

            mask = (
                    (geometry_b[:, 0] >= 0)
                    & (geometry_b[:, 0] < self.bev_dimension[0])
                    & (geometry_b[:, 1] >= 0)
                    & (geometry_b[:, 1] < self.bev_dimension[1])
                    & (geometry_b[:, 2] >= 0)
                    & (geometry_b[:, 2] < self.bev_dimension[2])
            )
            x_b = x_b[mask]
            geometry_b = geometry_b[mask]
            ranks = (
                    geometry_b[:, 0] * (self.bev_dimension[1] * self.bev_dimension[2])
                    + geometry_b[:, 1] * (self.bev_dimension[2])
                    + geometry_b[:, 2]
            )
            ranks_indices = ranks.argsort()
            x_b, geometry_b, ranks = x_b[ranks_indices], geometry_b[ranks_indices], ranks[ranks_indices]

            x_b, geometry_b = VoxelsSumming.apply(x_b, geometry_b, ranks)
            return geometry_b, x_b
        N = n * d * h * w
        for b in range(batch):
            flow_b = x[b]
            flow_geo = geometry[b]
            for t in range(s):
                if t != s - 1:
                    flow_geo_tmp = flow_geo[:t + 1]
                    rotation_b = rotation[b, t].view(1, 1, 1, 1, 1, 3, 3)  #
                    translation_b = translation[b, t].view(1, 1, 1, 1, 1, 3)
                    flow_geo_tmp = rotation_b.matmul(flow_geo_tmp.unsqueeze(-1)).squeeze(-1)
                    flow_geo_tmp += translation_b
                    flow_geo[:t + 1] = flow_geo_tmp

            bev_feature = torch.zeros((self.bev_dimension[2], self.bev_dimension[0], self.bev_dimension[1], c),
                                      device=flow_b.device)

            for t in range(s):
                x_b = flow_b[t].reshape(N, c)
                geometry_b = (
                        (flow_geo[t] - (self.bev_start_position - self.bev_resolution / 2.0)) / self.bev_resolution)
                geometry_b = geometry_b.view(N, 3).long()
                geometry_b, x_b = voxel_to_pixel(geometry_b, x_b)
                tmp_bev_feature = torch.zeros((self.bev_dimension[2], self.bev_dimension[0], self.bev_dimension[1], c),
                                              device=flow_b.device)
                tmp_bev_feature[geometry_b[:, 2], geometry_b[:, 0], geometry_b[:, 1]] = x_b
                bev_feature = tmp_bev_feature + tmp_bev_feature - bev_feature
                tmp_bev_feature = bev_feature.permute((0, 3, 1, 2))
                tmp_bev_feature = tmp_bev_feature.squeeze(0)
                output[b, t] = tmp_bev_feature

        return output