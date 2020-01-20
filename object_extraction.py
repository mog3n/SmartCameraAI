import os
from glob import glob
import cv2
import logging
import time

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import json


def create_folders():
    folders = ['detected_objects', 'faces']

    for folder in folders:
        folder_path = os.path.join(os.getcwd(), folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)


class ObjectExtraction:

    def __init__(self):
        # Initialization
        create_folders()
        logging.basicConfig(level=logging.INFO)
        self.db = None
        self.setup_db()

        # FB's Detectron Initialization
        setup_logger()
        self.cfg = get_cfg()
        model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        self.cfg.merge_from_file(model_zoo.get_config_file(model))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
        self.predictor = DefaultPredictor(self.cfg)
        self.index_to_class = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get('thing_classes')
        self.image_number = 0

    def process_faces(self, frame, video_path):
        pass

    def process_detectron(self, frame, video_path, video_index):
        outputs = self.predictor(frame)
        instances = outputs['instances']

        prediction_data = instances.get_fields()
        # print(prediction_data)
        for index in range(len(prediction_data['pred_classes'])):
            object_tensor = prediction_data['pred_classes'][index]
            object_name = self.index_to_class[object_tensor.item()]

            # ONLY ANNOTATE PEOPLE
            if object_name != "person":
                continue

            box_tensor = prediction_data['pred_boxes'][index].tensor
            x1 = int(box_tensor[0][0].item())
            y1 = int(box_tensor[0][1].item())
            x2 = int(box_tensor[0][2].item())
            y2 = int(box_tensor[0][3].item())

            # Save this area as a file
            object_roi = frame[y1:y2, x1:x2]
            # cv2.imshow(object_name, object_roi)

            # Save image to appropriate folder
            # make a folder named "object_name" if it doesnt exist
            path = os.path.join(os.getcwd(), "detected_objects", object_name)
            if not os.path.exists(path):
                os.mkdir(path)

            file_name = os.path.basename(video_path)
            image_path = os.path.join(os.getcwd(), "detected_objects", object_name,
                                      file_name + str(video_index)+".jpg")
            logging.info(image_path)
            # Save to path
            cv2.imwrite(image_path, object_roi)
            self.image_number += 1

            # Annotate
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)

        # cv2.imshow('Detectron 2 Custom', frame)

        # v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imshow('Detectron 2 Output', v.get_image()[:, :, ::-1])

    # Run this to extract features from a single frame
    def extract_features(self, frame, video_path, index):
        self.process_detectron(frame, video_path, index)

    def setup_db(self):
        db_path = os.path.join(os.getcwd(), 'annotated_videos.json')
        # Create new db file if it doesnt exist
        if not os.path.exists(db_path):
            db = {
                'annotated_videos': []
            }
            fs = open(db_path, 'w')
            fs.write(json.dumps(db))
            fs.close()
        else:
            # Otherwise, load one in
            fs = open(db_path, 'r')
            db = json.loads(fs.read())
            fs.close()

        self.db = db

    def save_db(self):
        db_path = os.path.join(os.getcwd(), 'annotated_videos.json')
        fs = open(db_path, 'w')
        fs.write(json.dumps(self.db))
        fs.close()
        logging.info("DB Saved")

    # Gets all footage from a folder and runs them through the ML extraction process
    def run(self):
        # get all video from folder
        videos_path = os.path.join(os.getcwd(), 'video', '*.mp4')
        videos = glob(videos_path)

        # Iterate through each video file path in the folder of videos
        for video_path in videos:
            index = 0
            video_name = os.path.basename(video_path)
            if video_name in self.db['annotated_videos']:
                # Do not annotate this video
                logging.info("Video already annotated")
                continue

            # Open video
            vid = cv2.VideoCapture(video_path)
            while vid.isOpened():
                ret, frame = vid.read()
                if not ret:
                    vid.release()
                    break

                # Analyze frame
                self.extract_features(frame, video_path, index)
                index += 1

                # if cv2.waitKey(1) == ord('q'):
                #     break

            # When annotations are complete, then we can add this file name to the list of annotated files
            self.db['annotated_videos'].append(video_name)
            self.save_db()

        logging.info("Done")


if __name__ == "__main__":
    oe = ObjectExtraction()
    while True:
        oe.run()
        time.sleep(5)
