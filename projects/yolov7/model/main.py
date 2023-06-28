import numpy as np
from PIL import Image
import cv2
from time import time
from TreeTopViewDetector_YOLOv7 import TreeTopViewDetector_YOLOv7

img_paths = [
    "../../data/xml/test/ABBY_009_2019_jpeg.rf.af45f1c3287a2fb03302ea14bad7b651.jpg",
    "../../data/xml/test/2018_TEAK_3_316000_4091000_image_446_jpeg.rf.593597e715a535970dba8ad811e8120d.jpg",
    "../../data/xml/test/BART_002_2019_jpeg.rf.758e5232f98c96f78d4de81e3b8b41a2.jpg",
    "../../data/xml/test/BART_037_2018_jpeg.rf.c5232edff33a75adf2c09fa7c99f2895.jpg",
    "../../data/xml/test/BART_037_2019_jpeg.rf.dd2344bb4197c5a8ae07ea86df849493.jpg",
    "../../data/xml/test/BART_044_2019_jpeg.rf.0b140c4f897ce1955ae906bd654c4272.jpg",
    "../../data/xml/test/CLBJ_006_2019_jpeg.rf.862b780249c80949097b4d6b9061e378.jpg"
]

imgs = [Image.open(i).convert("RGB") for i in img_paths]

# video = cv2.VideoCapture("../../../top-tree-detection/src/data/testing/video.mp4")

class_names = ["tree-top"]
# Step 1: Initialize model with the best available weights
# YOLOS model
WEIGHT_PATH = "./best.pt"

predictor = TreeTopViewDetector_YOLOv7(WEIGHT_PATH,class_names,threshold=0.8)

for i,img in enumerate(imgs):
    predictor.predict(img)
    img = predictor.draw(img)
    cv2.imshow(f"img-{i}",img)
    # cv2.imwrite(f"img-{i}.jpg",img)

cv2.waitKey(0)
