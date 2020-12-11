from yacs.config import CfgNode as CN

_C = CN()

_C.TRAIN = CN()
# The number of classes in dataset
_C.TRAIN.NUM_CLASSES = 20
# Learning rate
_C.TRAIN.BASE_LR = 0.00025
# Iteration in training
_C.TRAIN.MAX_ITER = 20000
# the path where log and checkpoint will be saved(Path not hand-craft created is acceptable).
_C.TRAIN.LOG_OUTPUT_PATH = "./log/output"
# training model config file
_C.TRAIN.CONFIG_FILE_PATH = "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
# pre-train model weight 
_C.TRAIN.MODEL_WEIGHTS_PATH = "detectron2://ImageNetPretrained/MSRA/R-101.pkl"

_C.TEST = CN()
# set a custom testing threshold
_C.TEST.SCORE_THRESH_TEST = 0.03
# pre-train model weight in testing 
_C.TEST.MODEL_WEIGHTS_PATH_TEST = "detectron2://ImageNetPretrained/MSRA/R-101.pkl"

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
  