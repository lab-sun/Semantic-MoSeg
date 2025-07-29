import os
import time
import socket
import torch 
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint , LearningRateMonitor
from timemoseg.config import get_parser, get_config
from timemoseg.data_util.dataloaders import prepare_dataloaders
from timemoseg.trainer_mos import TrainingModule 
import warnings
warnings.filterwarnings('ignore')

def main():
    args = get_parser().parse_args()
    cfg = get_config(args)

    trainloader, valloader = prepare_dataloaders(cfg)
    model = TrainingModule(cfg.convert_to_dict())
  
    save_dir = os.path.join(
        cfg.LOG_DIR, "moseg" )
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    

    checkpoint_callback = ModelCheckpoint(
        monitor='epoch_val_all_seg_iou_movingseg1',
        # 
        save_last=True,
        filename="{epoch:03d}_{epoch_val_all_seg_iou_movingseg1:.4f}",
        mode='max',
    ) # 

    trainer = pl.Trainer(
        gpus=cfg.GPUS,
        # accelerator='ddp',  #
        precision=cfg.PRECISION,
        sync_batchnorm=True,
        gradient_clip_val=cfg.GRAD_NORM_CLIP,
        max_epochs=cfg.EPOCHS,
        # weights_summary='full',
        logger=tb_logger,
        log_every_n_steps=cfg.LOGGING_INTERVAL,
        callbacks=[lr_monitor,checkpoint_callback]
    )
    trainer.fit(model, trainloader, valloader)


if __name__ == "__main__":
    main()
