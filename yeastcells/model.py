# -*- coding: utf-8 -*-
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def load_model(model_filename, seg_thresh=0.94, device='cpu'):
    '''
    Load and configure the Mask-RCNN model.
    Parameters
    ----------
    model_filename : str
        Path to model file containing weights, usually thisfilename
        ends with .pth .
    seg_thresh : float, optional
        Set the segmentation threshold score for the model to assign a 
        pixel to a cell region using the probability outcomes of 
        the predictor. The default is 0.94. Increasing the score may lead to 
        less segmentations, particularly of small buds, 
        while decrasing the score can lead to false positives.
    device : str, optional
        Can be set to a GPU or CPU depending on availability. 
        The default is 'cpu'.
    Returns
    -------
    predictor : DefaultPredictor
        Detectron2 predictor class object that takes one BGR image as input 
        and produces one output.
    '''
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = model_filename
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = seg_thresh
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.DEVICE=device
    predictor = DefaultPredictor(cfg)
    
    return predictor
