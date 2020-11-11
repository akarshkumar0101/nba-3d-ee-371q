import numpy as np
import json
import re
'''
numpy array

arr[framenumber][person_number][data_you_want]
where data_you_want can be 0=list for keypoint ordering (see below); 1=confidence score 
for the whole person and 2=list for box segmentation of the person

Keypoint ordering: 
// Result for COCO (17 body parts)
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

where {x,y,c} where c is the confidence score in the range [0,6]
'''

def get_numpy(jsonname):
    with open(jsonname) as f:
        jsondata = json.load(f)

    arr = [None] * len(jsondata)

    for feature in jsondata:
        if feature['category_id'] == 1: # is a person
            framenum = int(re.match("^.*_(.*)\..*$", feature['image_id']).group(1))
            adder = [feature['keypoints'], feature['score'], feature['box']]
            
            if not arr[framenum]:
                arr[framenum] = []
            arr[framenum].append(adder)


    cleaned = []
    for n in arr:
        if n: cleaned.append(n)

    np_out = np.asarray(cleaned)
    return np_out

