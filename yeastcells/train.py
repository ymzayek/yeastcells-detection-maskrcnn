import os
import umsgpack
from copy import deepcopy
from tqdm.cli import tqdm

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader


def create_model(
    model_path,
    device='cpu',
    data_workers=2,
    batch_size=2,
    learning_rate=0.00025,
    max_iter=20000,
    pretrained="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    tensorboard=None):
    cfg = get_cfg()
    
    if pretrained:
        cfg.merge_from_file(model_zoo.get_config_file(pretrained))

    cfg.DATASETS.TRAIN = ("yeast_cells_train",)
    cfg.DATASETS.TEST = ("yeast_cells_val",)
    cfg.DATALOADER.NUM_WORKERS = data_workers

    # Let training initialize from model zoo
    if pretrained:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(pretrained)

    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    if tensorboard:
        cfg.OUTPUT_DIR = tensorboard
    
    return cfg


def validate_labels(labels, path):
    """Check if all annotations have a simple polygon and
    filter out polygons with less than 3 points or of image
    files that do not exist"""
    for labels_ in labels.values():
        for label in labels_:
            for ann in label['annotations']:
                assert len(ann['segmentation']) == 1
                assert len(ann['segmentation'][0]) % 2 == 0

            label['annotations'] = [
                ann
                for ann in label['annotations']
                if len(ann['segmentation'][0]) >= 6
            ]
            assert len(label['annotations']) > 0
            label['file_name'] = path + '/' + label['file_name']

    for k in labels:
        labels[k] = [
            label for label in labels[k]
            if os.path.exists(label['file_name'])
        ]
    return labels


def register_data(path, prefix='yeast_cells_'):
    """Register all data sets as {prefix}_{setname}, i.e. yeast_cells_train"""
    assert (
        os.path.exists(f'{path}/labels.umsgpack') or
        os.path.exists(f'{path}/labels.json')), (
            "Labels not found, ensure either labels.umsgpack or labels.json "
            f"exists at {path}.")

    if os.path.exists(f'{path}/labels.umsgpack'):
      with open(f'{path}/labels.umsgpack', 'rb') as f:
        labels = umsgpack.unpack(f, encoding = "utf-8")
    else:
      with open(f'{path}/labels.json', 'r') as f:
        labels = json.load(f)

    labels = validate_labels(labels, path)

    DatasetCatalog.clear()
    for label in labels:
        DatasetCatalog.register(f"{prefix}{label}", lambda label_=label: labels[label_])
        MetadataCatalog.get(f"{prefix}{label}").set(thing_classes=["yeast_cell"])

    # yeast_cells_metadata = MetadataCatalog.get(f"{prefix}train")
    return labels


def train(config, data_path):
    """Train the Mask-RCNN for the given configuration and the given data"""
    register_data(data_path, prefix='yeast_cells_')
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(config)
    trainer.resume_or_load(resume=True)
    trainer.train()
    return trainer
