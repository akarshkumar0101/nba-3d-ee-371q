import numpy as np
import json
import re

'''
This script has multiple steps
1. Given a series of arguments, take in input image folder
2. generate output folder with all visualized images in, add alphapose-json to that folder
3. Take alphapose-json and convert to the numpy array as per requirements

'''


# do stuff here with command line stuff

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

with open('alphapose-results.json') as f:
    jsondata = json.load(f)

# output = np.empty((len(jsondata),10,3))
# output[34].add([[1,2,43,45,5],4545,[2,65,-435]])
# print(output[34])

arr = [None] * len(jsondata)


for feature in jsondata:
    if feature['category_id'] == 1: # is a person
        framenum = int(re.match("^.*_(.*)\..*$", feature['image_id']).group(1))
        adder = [feature['keypoints'], feature['score'], feature['box']]
        
        if not arr[framenum]:
            arr[framenum] = []
        arr[framenum].append(adder)

# verification
# print(len(arr))
# print(len(arr[1]))
# print(len(arr[1][0]))

np_out = np.asarray(arr)
print(np_out.shape)
print(np_out[1][0])
