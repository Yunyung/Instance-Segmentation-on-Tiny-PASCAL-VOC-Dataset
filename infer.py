# import some common libraries
import numpy as np
import os, json, cv2, random
from tqdm import tqdm

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
# if your dataset is in COCO format, this cell can be replaced by the following three lines:
from detectron2.data.datasets import register_coco_instances
# used to testing 
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode

# pycocotools used to form the format of submission json file
from pycocotools.coco import COCO
# used to encode segmentation mask and parser argument in cli
from utils import binary_mask_to_rle, default_argument_parser

# model config for this training/test
from config.default import get_cfg_defaults

register_coco_instances("my_dataset_test", {}, "dataset/coco/annotations/test.json", "dataset/coco/test2017")
dataset_metadata = MetadataCatalog.get("my_dataset_test")
# get the actual internal representation of the catalog stores information about the datasets and how to obtain them. The internal format uses one dict to represent the annotations of one image.
dataset_dicts = DatasetCatalog.get("my_dataset_test")
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


# training config (detectron2)
cfg = get_cfg()
cfg.merge_from_file(ep_config.TRAIN.CONFIG_FILE_PATH)
cfg.DATASETS.TRAIN = ()
cfg.DATASETS.TEST = () # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = ep_config.TRAIN.MODEL_WEIGHTS_PATH  # Let training initialize from model zoo(Pretrained on ImageNet)
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = ep_config.TRAIN.BASE_LR  # pick a good LR
cfg.SOLVER.MAX_ITER = ep_config.TRAIN.MAX_ITER  
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = ep_config.TRAIN.NUM_CLASSES  # 20 classes in this custom dataset

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = ep_config.TEST.MODEL_WEIGHTS_PATH_TEST  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = ep_config.TEST.SCORE_THRESH_TEST  # set a custom testing threshold

predictor = DefaultPredictor(cfg)


coco_dt = [] # Used to save list of dictionaries
# iterate each image in testing dataset and form the submmision file
for d in tqdm(dataset_dicts, ncols=100, mininterval=1, desc="Img Infer process"):
    img_id = d["image_id"]
    im = cv2.imread(d["file_name"])

    outputs = predictor(im)

    instances = outputs["instances"] # The num of instances detected by model in this image
    n_instances = len(instances) # len(instances) returns the number of instances

    categories = instances.pred_classes
    masks = instances.pred_masks
    scores = instances.scores

    masks = masks.permute(1, 2, 0).to("cpu").numpy()

    #print(categories)
    #print(masks[:,:,0])
    #print(scores)

    for i in range(n_instances): # Loop all instances
        # save information of the instance in a dictionary then append on coco_dt list
        pred = {} 
        pred['image_id'] = img_id # this imgid must be same as the key of test.json
        pred['category_id'] = int(categories[i]) + 1
        pred['segmentation'] = binary_mask_to_rle(masks[:, :, i]) # save binary mask to RLE, e.g. 512x512 -> rle
        pred['score'] = float(scores[i])
        coco_dt.append(pred)


with open("submission.json", "w") as f:
    json.dump(coco_dt, f)
