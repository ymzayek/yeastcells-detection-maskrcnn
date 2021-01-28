# -*- coding: utf-8 -*-
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

def load_model(model_path, seg_thresh=0.94, device='cpu'):
    log_prefix = '../output'
    cfg = get_cfg()
    cfg.OUTPUT_DIR = f'{log_prefix}/' 
    cfg.MODEL.WEIGHTS = f'{model_path}/model_final.pth'
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = seg_thresh
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.DEVICE=device #set GPU if available
    yeast_cells_metadata = MetadataCatalog.get("yeast_cells").set(thing_classes=["yeast_cell"])
    
    return DefaultPredictor(cfg)