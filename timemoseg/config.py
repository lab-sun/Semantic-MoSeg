import argparse
from pickle import FALSE
from fvcore.common.config import CfgNode as _CfgNode


def convert_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary."""
    _VALID_TYPES = {tuple, list, str, int, float, bool}
    if not isinstance(cfg_node, _CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print(
                'Key {} with value {} is not a valid type; valid types: {}'.format(
                    '.'.join(key_list), type(cfg_node), _VALID_TYPES
                ),
            )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


class CfgNode(_CfgNode):
    """Remove once https://github.com/rbgirshick/yacs/issues/19 is merged."""

    def convert_to_dict(self):
        return convert_to_dict(self)


CN = CfgNode

_C = CN()
_C.LOG_DIR = 'tensorboard_logs'
_C.TAG = 'default'

_C.GPUS = [2]  
_C.PRECISION =  16
_C.BATCHSIZE = 6 #4 # 8
_C.EPOCHS = 50

_C.N_WORKERS = 8
_C.VIS_INTERVAL = 5000
_C.LOGGING_INTERVAL = 500

_C.PRETRAINED = CN()
_C.PRETRAINED.LOAD_WEIGHTS =  False # True
_C.PRETRAINED.PATH = '' # ''
_C.FILETER_SCENE_PATH = 'onlytrain.txt'  #  
_C.TRAIN_SCENE_PATH = 'txt/train_550.txt'
_C.VAL_SCENE_PATH = 'txt/val_150.txt'  #150
_C.TEST_SCENE_PATH = 'txt/test_150.txt'   #150
_C.TEST_SAVE_PATH = 'output_save'   #150

_C.DATASET = CN()
_C.DATASET.DATAROOT =   'dataset'
_C.DATASET.VERSION =   'trainval' # 'trainval'
_C.DATASET.NAME = 'nuscenes'
_C.DATASET.MAP_FOLDER =  'dataset' # 'v10-mini' #' 'data'
_C.DATASET.IGNORE_INDEX = 255  #  
_C.DATASET.FILTER_INVISIBLE_VEHICLES = True  #  
_C.DATASET.SAVE_DIR = 'datas'

_C.TIME_RECEPTIVE_FIELD = 2  # how many frames  
_C.POSE_USE =True   

_C.IMAGE = CN()
_C.IMAGE.FINAL_DIM = (224, 480)
_C.IMAGE.RESIZE_SCALE = 0.3
_C.IMAGE.TOP_CROP = 46

_C.IMAGE.ORIGINAL_HEIGHT = 900  # Original input RGB camera height
_C.IMAGE.ORIGINAL_WIDTH = 1600  # Original input RGB camera width
_C.IMAGE.NAMES = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

_C.IMAGE.USEDEPTH = True


_C.MOVING_SEG = CN()
_C.MOVING_SEG.ENABLED = True
_C.MOVING_SEG.WEIGHTS = [1.0, 2.0]
_C.MOVING_SEG.TOP_K_RATIO = 0.25

_C.LIFT = CN()   
_C.LIFT.X_BOUND =  [-50.0, 50.0, 0.5]      
_C.LIFT.Y_BOUND =   [-50.0, 50.0, 0.5]  # 
_C.LIFT.Z_BOUND = [-10.0, 10.0, 20.0]  #  
_C.LIFT.D_BOUND =   [2.0, 50.0, 1.0]    
 

_C.EGO = CN()
_C.EGO.WIDTH = 1.85
_C.EGO.HEIGHT = 4.084
_C.MODEL = CN()
_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.DOWNFACTOR = 8
_C.MODEL.ENCODER.NAME = 'efficientnet-b4' # 'efficientnet-b4'
_C.MODEL.ENCODER.OUT_CHANNELS = 128
_C.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION = True


_C.MODEL.TEMPORAL_MODEL = CN()
_C.MODEL.TEMPORAL_MODEL.NAME = 'temporal_block'  # type of temporal model
_C.MODEL.TEMPORAL_MODEL.START_OUT_CHANNELS = 128
_C.MODEL.TEMPORAL_MODEL.EXTRA_IN_CHANNELS = 0
_C.MODEL.TEMPORAL_MODEL.INBETWEEN_LAYERS = 0
_C.MODEL.TEMPORAL_MODEL.PYRAMID_POOLING = False#True
_C.MODEL.TEMPORAL_MODEL.INPUT_EGOPOSE = False



_C.MODEL.DECODER = CN()
_C.MODEL.BN_MOMENTUM = 0.1
_C.SEMANTIC_SEG = CN()
_C.SEMANTIC_SEG.VEHICLE = CN()
_C.SEMANTIC_SEG.VEHICLE.ENABLED = True  #  
_C.SEMANTIC_SEG.VEHICLE.WEIGHTS = [1.0, 2.0]
_C.SEMANTIC_SEG.VEHICLE.USE_TOP_K = True  #  
_C.SEMANTIC_SEG.VEHICLE.TOP_K_RATIO = 0.25

_C.SEMANTIC_SEG.HDMAP = CN()
_C.SEMANTIC_SEG.HDMAP.ENABLED = False # True
_C.SEMANTIC_SEG.HDMAP.ELEMENTS = ['lane_divider', 'drivable_area']

_C.OPTIMIZER = CN()
_C.OPTIMIZER.LR =    1e-3
_C.OPTIMIZER.WEIGHT_DECAY =  1e-7
_C.GRAD_NORM_CLIP = 5


_C.TRANS  = CN()
_C.TRANS.DIM_LAST = 64
_C.TRANS.HEADS: 4
_C.TRANS.DIM_HEAD: 32
# _C.TRANS.QKV_BIAS: True
# _C.TRANS.SKIP: True
# _C.TRANS.ON_IMAGE_FEATURES: False
# _C.TRANS.IMAGE_HEIGHT: 224
# _C.TRANS.IMAGE_WIDTH: 480
# _C.TRANS.SIGMA: 1.0
# _C.TRANS.BEV_HEIGHT: 200
# _C.TRANS.BEV_WIDTH: 200
# _C.TRANS.H_METERS: 100.0
# _C.TRANS.W_METERS: 100.0
# _C.TRANS.OFFSET: 0





def get_parser():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config-file', default='', metavar='FILE',)
    parser.add_argument(
        'opts', help='', default=None, nargs=argparse.REMAINDER,
    )
    return parser


def get_config(args=None, cfg_dict=None):

    cfg = _C.clone()

    if cfg_dict is not None:
        tmp = CfgNode(cfg_dict)
        cfg.merge_from_other_cfg(tmp)

    if args is not None:
        if args.config_file:
            cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        #  
    return cfg
