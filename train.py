# import some common libraries
import numpy as np
import os, json, cv2, random

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
from detectron2.data.datasets import register_coco_instances
# used to traning 
from detectron2.engine import DefaultTrainer

# model config for this training/test
from config.default import get_cfg_defaults
from utils import default_argument_parser
register_coco_instances("my_dataset_train", {}, "dataset/coco/annotations/pascal_train.json", "dataset/coco/train2017")
dataset_metadata = MetadataCatalog.get("my_dataset_train")
# get the actual internal representation of the catalog stores information about the datasets and how to obtain them. The internal format uses one dict to represent the annotations of one image.
dataset_dicts = DatasetCatalog.get("my_dataset_train")
print(dataset_metadata)
# print(dataset_dicts)

# parse argument from cli
args = default_argument_parser().parse_args()

# configuration 
ep_config = get_cfg_defaults() 
if args.experiment_file is not None:
    ep_config.merge_from_file(args.experiment_file) # configuration for this experiment
ep_config.freeze()
print(ep_config)

# training config
cfg = get_cfg()
cfg.merge_from_file(ep_config.TRAIN.CONFIG_FILE_PATH)
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = () # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = ep_config.TRAIN.MODEL_WEIGHTS_PATH  # Let training initialize from model zoo(Pretrained on ImageNet
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = ep_config.TRAIN.BASE_LR  # pick a good LR
cfg.SOLVER.MAX_ITER = ep_config.TRAIN.MAX_ITER  
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = ep_config.TRAIN.NUM_CLASSES  # 20 classes in this custom dataset

os.makedirs(ep_config.TRAIN.LOG_OUTPUT_PATH, exist_ok=True) # create folder if not exist
cfg.OUTPUT_DIR = ep_config.TRAIN.LOG_OUTPUT_PATH # set training log output path

trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
    