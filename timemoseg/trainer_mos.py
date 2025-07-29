import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy
from timemoseg.config import get_config
from torch.optim.lr_scheduler import MultiStepLR

from timemoseg.models.moseg_model import MOSEG
from timemoseg.utils.util import compute_results
from timemoseg.losses_func import SegmentationLoss, Lovasz_softmax, get_depth_loss
from timemoseg.metrics import IoUmetric
from timemoseg.utils.geometry import coord_features
# from timemoseg.utils.visualize_removemapman import visualise_output
#
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


class TrainingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        #
        self.hparams = hparams
        cfg = get_config(cfg_dict=self.hparams)
        self.cfg = cfg
        self.n_classes = len(self.cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS)

        assert self.cfg.LIFT.X_BOUND[1] > 0 and self.cfg.LIFT.Y_BOUND[1] > 0
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])
        self.n_eye_steps = self.cfg.TIME_RECEPTIVE_FIELD  #
        # Model
        self.model = MOSEG(cfg)

        self.losses_fn = nn.ModuleDict()
        if self.cfg.SEMANTIC_SEG.VEHICLE.ENABLED:
            # movable segmentation
            self.losses_fn['segmentation'] = SegmentationLoss(
                class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS),
                use_top_k=self.cfg.SEMANTIC_SEG.VEHICLE.USE_TOP_K,
                top_k_ratio=self.cfg.SEMANTIC_SEG.VEHICLE.TOP_K_RATIO,
            )

            self.model.segmentation_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

            """timestamp Iou"""
            for s in range(self.n_eye_steps):
                exec("self.metric_vehicle_val_" + str(s) + "=IoUmetric(self.n_classes)")  #
            #
        if self.cfg.IMAGE.USEDEPTH:
            self.model.depths_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # movignseg
        if self.cfg.MOVING_SEG.ENABLED:  # true
            self.losses_fn['moving_seg'] = SegmentationLoss(
                class_weights=torch.Tensor(self.cfg.MOVING_SEG.WEIGHTS),
                use_top_k=True,
                top_k_ratio=self.cfg.MOVING_SEG.TOP_K_RATIO,
            )
            # self.loss_ls = Lovasz_softmax()
            # self.losses_fn['lsoftmax'] = Lovasz_softmax()
            self.model.movingseg_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            """iou timestemp"""
            for s in range(self.n_eye_steps):  #
                exec("self.metric_moving_val_" + str(s) + "=IoUmetric(self.n_classes)")

        self.training_step_count = 0

    def shared_step(self, batch, is_train):
        image = batch['image']  #
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        vehicle_egomotion = batch['egomotion']
        #
        labels = self.prepare_labels(batch)
        # Forward
        output = self.model( image, intrinsics, extrinsics, vehicle_egomotion)
        # Loss 
        loss = {}
        dict_confusion_matrix = {}
        seg_dict_confusion_matrix = {}  # movable
        if is_train:
            # segmentation
            if self.cfg.SEMANTIC_SEG.VEHICLE.ENABLED:
                segmentation_factor = 1 / (2 * torch.exp(self.model.segmentation_weight))
                loss['segmentation'] = segmentation_factor * self.losses_fn['segmentation'](
                    output['segmentation'], labels['segmentation'], self.model.receptive_field
                )

                loss['segmentation_uncertainty'] = 0.5 * self.model.segmentation_weight

            """moving loss"""
            if self.cfg.MOVING_SEG.ENABLED:
                movingseg_factor = 1 / (2 * torch.exp(self.model.movingseg_weight))
                loss['moving_seg'] = movingseg_factor * self.losses_fn['moving_seg'](
                    output['moving_seg'], labels['moving_seg'], self.model.receptive_field
                )
                loss['movingseg_uncertainty'] = 0.5 * self.model.movingseg_weight
                ###############################################

            if self.cfg.IMAGE.USEDEPTH:
                #
                depths_factor = 1 / (2 * torch.exp(self.model.depths_weight))
                loss['depths'] = depths_factor * get_depth_loss(depth_labels=labels['depths'],
                                                                depth_preds=output['depth_prob'],
                                                                depth_channels=self.model.depth_channels)
                loss['depths_uncertainty'] = 0.5 * self.model.depths_weight
            """moving loss"""

            output = {**output, }
        # Metrics
        else:
            n_present = self.model.receptive_field  # 3
            if self.cfg.SEMANTIC_SEG.VEHICLE.ENABLED:
                #
                seg_prediction = output['segmentation'].detach()
                #
                seg_prediction = torch.argmax(seg_prediction, dim=2, keepdim=True)
                # on
                for s in range(n_present):  # time_stemp
                    label = labels['segmentation'][:, s: s + 1].squeeze().flatten()
                    #
                    exec("self.metric_vehicle_val_" + str(
                        s) + "(seg_prediction[:,s: s+1 ], labels['segmentation'][:,s: s+1])")

                    seg_dict_confusion_matrix[s] = (
                        self.get_step_cm_yuyi(seg_prediction, labels, s)
                    )

            # moving
            if self.cfg.MOVING_SEG.ENABLED:
                # compute iou
                movingseg_prediction = output['moving_seg'].detach()
                movingseg_prediction = torch.argmax(movingseg_prediction, dim=2, keepdim=True)
                # only
                for s in range(n_present):  # time_stamp
                    label = labels['moving_seg'][:, s: s + 1].squeeze().flatten()
                    #
                    exec("self.metric_moving_val_" + str(
                        s) + "(movingseg_prediction[:,s: s+1 ], labels['moving_seg'][:,s: s+1])")

                    dict_confusion_matrix[s] = (
                        self.get_step_confusion_matrix(movingseg_prediction, labels, s)
                    )

            output = {**output, }

        return output, labels, loss, dict_confusion_matrix, seg_dict_confusion_matrix  #

    def prepare_labels(self, batch):
        labels = {}
        vehicle_egomotion = batch['egomotion']

        if self.cfg.SEMANTIC_SEG.VEHICLE.ENABLED:
            segmentation_labels = batch['segmentation']
            segmentation_labels = coord_features(
                segmentation_labels[:, :self.model.receptive_field].float(),
                vehicle_egomotion[:, :self.model.receptive_field],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).long().contiguous()  #
            labels['segmentation'] = segmentation_labels

        #  only for visualize
        hdmap_labels = batch['hdmap']  #
        labels['hdmap'] = hdmap_labels[:, self.model.receptive_field - 1].long().contiguous()

        # depth
        if self.cfg.IMAGE.USEDEPTH:
            depths = batch['depths']
            depth_labels = self.get_downsampled_gt_depth(depths)
            labels['depths'] = depth_labels

        if self.cfg.MOVING_SEG.ENABLED:
            movingseg_labels = batch['moving_seg']
            movingseg_labels = coord_features(
                movingseg_labels[:, :self.model.receptive_field].float(),
                vehicle_egomotion[:, :self.model.receptive_field],
                mode='nearest', spatial_extent=self.spatial_extent, ).long().contiguous()
            labels['moving_seg'] = movingseg_labels

        return labels

    # def visualise(self, labels, output, batch_idx, prefix='train'):
    #     visualisation_video = visualise_output(labels, output, self.cfg)
    #     name = f'{prefix}_outputs'
    #     if prefix == 'val':
    #         name = name + f'_{batch_idx}'
    #     self.logger.experiment.add_video(name, visualisation_video, global_step=self.training_step_count, fps=2)

    def training_step(self, batch, batch_idx):
        output, labels, loss, _, _ = self.shared_step(batch, True)
        self.training_step_count += 1
        for key, value in loss.items():
            self.logger.experiment.add_scalar('step_train_loss_' + key, value, global_step=self.training_step_count)
        # if self.training_step_count % self.cfg.VIS_INTERVAL == 0:
        #     self.visualise(labels, output, batch_idx, prefix='train')
        return sum(loss.values())

    def validation_step(self, batch, batch_idx):
        output, labels, _, dict_confusion_matrix, seg_dict_confusion_matrix = self.shared_step(batch, False)

        if self.cfg.SEMANTIC_SEG.VEHICLE.ENABLED:
            for s in range(self.n_eye_steps):
                exec("scores_" + str(s) + "=" + "self.metric_vehicle_val_" + str(s) + ".compute()")
                exec("self.log('step_val_seg_iou_dynamic{}'.format(s)," + "scores_" + str(
                    s) + "[1])")  # tensor([0.9785, 0.0000], device='cuda:3') 因为是2类，第一个类别是背景

        for s in range(self.n_eye_steps):
            exec("scores_" + str(s) + "=" + "self.metric_moving_val_" + str(s) + ".compute()")
            exec("self.log('step_val_seg_iou_movingseg{}'.format(s)," + "scores_" + str(s) + "[1])")

        # if batch_idx == 0:
        #     self.visualise(labels, output, batch_idx, prefix='val')
        return {"mos": dict_confusion_matrix, "segmentation": seg_dict_confusion_matrix}

    def get_step_confusion_matrix(self, movingseg_prediction, labels, s):
        label = labels['moving_seg'][:, s: s + 1].cpu().numpy().squeeze().flatten()
        prediction = movingseg_prediction[:, s: s + 1].cpu().numpy().squeeze().flatten()
        con_matrix = confusion_matrix(
            y_true=label, y_pred=prediction, labels=[0, 1]
        )
        return con_matrix  # step hunxiaojuzhen

    def get_step_cm_yuyi(self, seg_prediction, labels, s):
        label = labels['segmentation'][:, s: s + 1].cpu().numpy().squeeze().flatten()
        prediction = seg_prediction[:, s: s + 1].cpu().numpy().squeeze().flatten()
        con_matrix = confusion_matrix(
            y_true=label, y_pred=prediction, labels=[0, 1]
        )
        return con_matrix

    def get_downsampled_gt_depth(self, gt_depths):

        B, t, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * t * N,
            H // self.model.encoder_downfactor,
            self.model.encoder_downfactor,
            W // self.model.encoder_downfactor,
            self.model.encoder_downfactor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.model.encoder_downfactor * self.model.encoder_downfactor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * t * N, H // self.model.encoder_downfactor,
                                   W // self.model.encoder_downfactor)  #

        gt_depths = (gt_depths -
                     (self.cfg.LIFT.D_BOUND[0] - self.cfg.LIFT.D_BOUND[2])) / self.cfg.LIFT.D_BOUND[2]
        gt_depths = torch.where(
            (gt_depths < self.model.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.model.depth_channels + 1).view(
            -1, self.model.depth_channels + 1)[:, 1:]

        return gt_depths.float()

    def shared_epoch_end(self, step_outputs, is_train):
        if is_train == "val":  #
            moving_cf = [output["mos"] for output in step_outputs]
            movable_cf = [output["segmentation"] for output in step_outputs]
            #
            if self.cfg.SEMANTIC_SEG.VEHICLE.ENABLED:
                for s in range(self.n_eye_steps):
                    agg_confusion_matrix = numpy.zeros((self.n_classes, self.n_classes))
                    for dict_confusion_matrix in movable_cf:
                        agg_confusion_matrix += dict_confusion_matrix[s]
                    # recall precision  iou
                    precision, recall, IoU = compute_results(agg_confusion_matrix)

                    exec("scores_" + str(s) + "=" + "self.metric_vehicle_val_" + str(s) + ".compute()")
                    exec("self.log('epoch_val_all_seg_iou_dynamic{}'.format(s)," + "scores_" + str(s) + "[1])")
                    exec("print('epoch_val_all_seg_iou_dynamic{}'.format(s)," + "scores_" + str(s) + "[1])")
                    exec("self.metric_vehicle_val_" + str(s) + "." + "reset()")

            # car msy !!!!! moving seg
            if self.cfg.MOVING_SEG.ENABLED:  #
                for s in range(self.n_eye_steps):
                    agg_confusion_matrix = numpy.zeros((self.n_classes, self.n_classes))
                    for dict_confusion_matrix in moving_cf:
                        #
                        agg_confusion_matrix += dict_confusion_matrix[s]
                    # recall precision  iou compute
                    precision, recall, IoU = compute_results(agg_confusion_matrix)

                    #
                    exec("scores_" + str(s) + "=" + "self.metric_moving_val_" + str(s) + ".compute()")
                    exec("self.log('epoch_val_all_seg_iou_movingseg{}'.format(s)," + "scores_" + str(s) + "[1])")
                    exec("print('epoch_val_all_seg_iou_movingseg{}'.format(s)," + "scores_" + str(s) + "[1])")
                    exec("self.metric_moving_val_" + str(s) + "." + "reset()")

        #
        if self.cfg.SEMANTIC_SEG.VEHICLE.ENABLED:
            self.logger.experiment.add_scalar('epoch_segmentation_weight',
                                              1 / (2 * torch.exp(self.model.segmentation_weight)),
                                              global_step=self.training_step_count)
        if self.cfg.MOVING_SEG.ENABLED:
            self.logger.experiment.add_scalar('epoch_movingseg_weight',
                                              1 / (2 * torch.exp(self.model.movingseg_weight)),
                                              global_step=self.training_step_count)
        if self.cfg.IMAGE.USEDEPTH:
            self.logger.experiment.add_scalar('epoch_depths_weight', 1 / (2 * torch.exp(self.model.depths_weight)),
                                              global_step=self.training_step_count)

    def training_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, "train")

    def validation_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, "val")
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        params = self.model.parameters()

        optimizer = torch.optim.AdamW(
            params, lr=self.cfg.OPTIMIZER.LR, weight_decay=self.cfg.OPTIMIZER.WEIGHT_DECAY
        )
        #
        scheduler = MultiStepLR(optimizer, [15, 19])  #
        return [[optimizer], [scheduler]]



