import detectron2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import cv2
import requests
import numpy as np

cfg = get_cfg()
cfgFile = '/home/dirac/miniconda3/envs/detectron/lib/python3.6/site-packages/detectron2/'