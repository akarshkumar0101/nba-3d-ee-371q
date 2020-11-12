import os
import argparse
'''
This script has multiple steps
1. Given a series of arguments, take in input image folder
2. generate output folder with all visualized images in, add alphapose-json to that folder
3. Take alphapose-json and convert to the numpy array as per requirements

'''


# do stuff here with command line stuff
parser = argparse.ArgumentParser(description='pass in the name of the input dir')
parser.add_argument('input')
args = parser.parse_args()


print(args.input)
cmd = f'python alphapose/Alphapose/scripts/demo_inference.py --cfg alphapose/Alphapose/configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml --checkpoint alphapose/Alphapose/pretrained_models/fast_421_res152_256x192.pth --indir basketball/{args.input} --outdir basketball/{args.input}-alphapose --vis --save_img --save_video'

print(cmd)
os.system(cmd)
