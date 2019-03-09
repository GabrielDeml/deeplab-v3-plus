
import cv2
import numpy as np
import argparse
import os
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument("--img_dir", type=str, default='./dataset/export/train/labels/')
parser.add_argument("--out_dir", type=str, default="./dataset/export/train/labelsAug/")

args = parser.parse_args()

kernel = np.ones((5,5), np.uint8)

palette = {(0,   0,   0) : 0 ,
         (0,  0, 255) : 1 ,
         (255,  0,  0) : 2,
           (255,255,  0) : 3,
           (  0,255,  0) : 4,
           (255,  0,255) : 5,
           (  0,255,255) : 6,
           (255,  0,153) : 7,
           (153,  0,255) : 8,
           (  0,153,255) : 9,
           (153,255,  0) : 10,
           (255,153,  0) : 11,
           (  0,255,153) : 12,
           (  0,153,153) : 13
          }

def convert_from_color_segmentation(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

print("Processing: " + args.img_dir + "\nPutting them: " + args.out_dir)

for name in os.listdir(args.img_dir):
    print("Reading: " + args.img_dir + name)
    # Read img
    img = cv2.imread(args.img_dir + name, 1)

    #Find edges
    edges = cv2.Canny(img, 100, 200)
    img_dilation = cv2.dilate(edges, kernel, iterations=1)

    cColor = convert_from_color_segmentation(img)


    fImage = cv2.add(edges, cColor)

    #Write it to a file
    cv2.imwrite(args.out_dir + name, fImage)
