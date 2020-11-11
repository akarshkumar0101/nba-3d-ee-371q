import detectron2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import cv2
import numpy as np


image = cv2.imread('/home/dirac/basketball/all_frames/frame_00004.png', cv2.IMREAD_COLOR) # pick an option with the ball visible

cfg = get_cfg()
cfgFile = '/home/dirac/miniconda3/envs/detectron/lib/python3.6/site-packages/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
cfg.merge_from_file(cfgFile)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"

predictor = DefaultPredictor(cfg)

output = predictor(image)
print(output)

# visualize this prediction
v = Visualizer(image[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(output["instances"].to("cpu"))
cv2.imshow('visualize', out.get_image()[:,:,::-1])
cv2.waitKey(0)

