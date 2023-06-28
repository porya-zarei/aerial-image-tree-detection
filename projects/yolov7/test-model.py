import torch
import cv2
import numpy as np
from PIL import Image

# Model
model = torch.hub.load('WongKinYiu/yolov7','custom',"./model/best.pt",force_reload=True)  # or yolov5m, yolov5l, yolov5x, custom

# Images
img_path = '../data/xml/test/2018_SJER_3_252000_4113000_image_323_jpeg.rf.ed36edac98408f1acb5e866d7e4e45d8.jpg'  # or file, Path, PIL, OpenCV, numpy, list
image = Image.open(img_path)
# Inference
results = model(image)

# print(results)

print(f"result => {results.xyxy[0]}")  # im predictions (pandas)
# show image
image.show()