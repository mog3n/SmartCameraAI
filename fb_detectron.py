from __future__ import print_function
from detectron2.utils.logger import setup_logger

# Common libs
import torch
import torchvision
import cv2
import numpy as np
import logging
import time
import random
import detectron2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

setup_logger()  # From detectron2
logging.basicConfig(level=logging.INFO)
logging.info("Started")

# SETUP FOR DETECTRON
cfg = get_cfg()

# Model to use
MODEL = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# MODEL = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"

cfg.merge_from_file(model_zoo.get_config_file(MODEL))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
# Find a model from detectron's model zooimport
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)
predictor = DefaultPredictor(cfg)

index_to_class = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get('thing_classes')


def process_detectron(frame):
    outputs = predictor(frame)
    instances = outputs['instances']

    should_show_frame = False
    prediction_data = instances.get_fields()
    for index in range(len(prediction_data['pred_classes'])):
        object_tensor = prediction_data['pred_classes'][index]
        object_index = object_tensor.item()
        # Get actual object name
        object_name = index_to_class[object_index]

        if object_name == "person":
            should_show_frame = True

    if should_show_frame:
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow('Detectron 2 Output', v.get_image()[:, :, ::-1])


def main():
    vid = cv2.VideoCapture('video3.mp4')
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            logging.info("Couldn't read frame")

        # cv2.imshow("Original", frame)
        process_detectron(frame)

        if cv2.waitKey(1) == ord('q'):
            break
        elif cv2.waitKey(1) == ord('w'):
            time.sleep(1)


if __name__ == "__main__":
    main()