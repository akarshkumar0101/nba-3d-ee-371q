import numpy as np
import json
import re
import os

"""
returns kp, confs, scores, boxes.
These are the keypoints, confidences/keypoint, confidence/body, and bounding boxes respectively.

kp has indices [view_idx][frame_idx][person_idx][17, 2].
confs has indices [view_idx][frame_idx][person_idx][17].
scores has indices [view_idx][frame_idx][person_idx].
confs has indices [view_idx][frame_idx][person_idx][2, 2].

The location values are in (x, y), NOT (y, x) coordinates for an image.

Keypoint ordering: 
Result for COCO (17 body parts)
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
"""
def parse_alphapose(data_root, num_frames):
    dirs = []
    for d in os.listdir(data_root):
        if re.match('view_[0-9]_alphapose', d):
            dirs.append(d)
    dirs.sort()
    kp, confs, scores, boxes = [], [], [], []

    for d in dirs:
        kp_view, confs_view, scores_view, boxes_view = parse_alphapose_json_file(f'{data_root}/{d}/alphapose-results.json', num_frames)
        kp.append(kp_view)
        confs.append(confs_view)
        scores.append(scores_view)
        boxes.append(boxes_view)
    return kp, confs, scores, boxes

def parse_alphapose_json_file(json_file, num_frames):
    with open(json_file) as f:
            json_data = json.load(f)
    return parse_alphapose_json(json_data, num_frames)

def parse_alphapose_json(json_data, num_frames):
    kp = [[] for _ in range(num_frames)]
    confs = [[] for _ in range(num_frames)]
    scores = [[] for _ in range(num_frames)]
    boxes = [[] for _ in range(num_frames)]

    for feature in json_data:
        if feature['category_id'] == 1: # is a person
            frame_idx = int(re.match("^.*_(.*)\..*$", feature['image_id']).group(1)) #REGEXXXXXXX
            kp_conf_person = np.array(feature['keypoints']).reshape(17, 3)
            box_person = np.array(feature['box']).reshape(2, 2)
            box_person[1] += box_person[0] # get the absolute value of second point, not just offset.
            
            kp[frame_idx].append(kp_conf_person[:, :2])
            confs[frame_idx].append(kp_conf_person[:, 2])
            scores[frame_idx].append(np.array(feature['score']))
            boxes[frame_idx].append(box_person)
    return kp, confs, scores, boxes
