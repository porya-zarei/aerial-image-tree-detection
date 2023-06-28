import numpy as np
from transformers import pipeline
from PIL import Image
import cv2
from time import time
from TreeTopViewDetector_YOLOS import TreeTopViewDetector_YOLOS
img_paths = [
    "../../../top-tree-detection/src/data/testing/3.jpg",
    "../../../top-tree-detection/src/data/testing/4.jpg",
    "../../../top-tree-detection/src/data/testing/5.jpg",
    "../../../top-tree-detection/src/data/testing/6.jpg",
    "../../../top-tree-detection/src/data/testing/7.jpg",
    "../../../top-tree-detection/src/data/testing/ABBY_065_2019_2.jpeg",
    "../../../top-tree-detection/src/data/testing/ABBY_029_2019.jpeg"
]
imgs = [Image.open(i).convert("RGB") for i in img_paths]

class_names = ["tree-top"]
# Step 1: Initialize model with the best available weights
# YOLOS model
model_path = "../deploy/"

predictor = TreeTopViewDetector_YOLOS(model_path,class_names,threshold=0.8)

for i in range(len(imgs)):
    img = imgs[i]
    predictor.predict(img)
    img = predictor.draw(img)
    cv2.imshow(f"img-{i}",img)
cv2.waitKey(0)
