import torch
import torch.utils.data
from nuscenes.nuscenes import NuScenes

from timemoseg.data_util.nuscenes_moseg import  MOSDataset 


def prepare_dataloaders(cfg, return_dataset=False, test =False):
    if cfg.DATASET.NAME == 'nuscenes':
        #  
        dataroot = cfg.DATASET.DATAROOT
        nusc = NuScenes(version='v1.0-{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=False)
        traindata = MOSDataset(nusc, 0, cfg) # train
        valdata = MOSDataset(nusc, 1, cfg)  # val
        if test :
            testdata = MOSDataset(nusc, 2, cfg)  # test

        nworkers = cfg.N_WORKERS
        trainloader = torch.utils.data.DataLoader(
            traindata, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=nworkers, pin_memory=True, prefetch_factor=2, drop_last=True
        )
        valloader = torch.utils.data.DataLoader(
            valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, prefetch_factor=2, drop_last=False)
        if test:
            testloader = torch.utils.data.DataLoader(
            testdata, batch_size=1, shuffle=False, num_workers=nworkers, pin_memory=True, prefetch_factor=2, drop_last=False)

    if return_dataset:
        return trainloader, valloader, traindata, valdata,
    elif test:
        return testloader
    else:
        return trainloader, valloader