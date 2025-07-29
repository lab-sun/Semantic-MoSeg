import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback

class Meng_callback( Callback):
    def on_init_start( self, trainer):
        print( "starting")
    def on_init_end( self, trainer):
        print( "init now")

    def on_train_start( self,trainer, pl_module ):
        pl_module.previous_label_pre_train= []
        pl_module.current_epoch_pre_train =[]
        pl_module.previous_label_pre_val= []
    def on_validation_start( self,trainer, pl_module ):
        
        pl_module.current_epoch_pre_val =[]

    def on_train_end( self,trainer, pl_module):
        del pl_module.previous_label_pre_train 
        del pl_module.current_epoch_pre_train
        
    # def on_validation_end( self,trainer, pl_module):
        # del pl_module.previous_label_pre_val 
        # del pl_module.current_epoch_pre_val

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.previous_label_pre_train= pl_module.current_epoch_pre_train
        pl_module.current_epoch_pre_train =[]

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.previous_label_pre_val= pl_module.current_epoch_pre_val 
        pl_module.current_epoch_pre_val =[]