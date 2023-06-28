import numpy as np
from transformers import pipeline
from PIL import Image
import cv2
from time import time

def load_model(model_dir,threshold=0.5):
    pipe = pipeline("object-detection", model=model_dir, threshold=threshold)
    return pipe
def get_label(classes,box_data):
    label = box_data["label"]
    index = label.split("_")[-1]
    return classes[int(index)-1]
def draw_box(img,box_data,color=(0,255,00),thickness=2):
    xmin = box_data["box"]["xmin"]
    ymin = box_data["box"]["ymin"]
    xmax = box_data["box"]["xmax"]
    ymax = box_data["box"]["ymax"]
    label = get_label(class_names,box_data)
    score = box_data["score"]
    cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color,thickness)
    cv2.putText(img,f"{label}({score:0.3f})",(xmin,ymin),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
    return img
def draw_boxes(img,box_data):
    for box in box_data:
        img = draw_box(img,box)
    return img

img = Image.open("../../../../data/test/2018_SJER_3_252000_4106000_image_234_jpeg.rf.5bf0ffaf21b4ca17d380f3a79550f4c8.jpg",)
class_names = ["tree-top"]
# Step 1: Initialize model with the best available weights
weights = "../model_weights.pth"
# YOLOS model
model_path = "../"

model = load_model(model_path)

# Step 2: Initialize the inference transforms

model.model.eval()


start_prediction_time = time()
res = model(img)
end_prediction_time = time()
"""
[
    {'score': 0.13099995255470276, 'label': 'LABEL_1', 'box': {'xmin': 83, 'ymin': 151, 'xmax': 165, 'ymax': 246}}, # box_data
    {'score': 0.8488465547561646, 'label': 'LABEL_1', 'box': {'xmin': 315, 'ymin': 138, 'xmax': 409, 'ymax': 228}},
    {'score': 0.40752533078193665, 'label': 'LABEL_1', 'box': {'xmin': 43, 'ymin': 26, 'xmax': 74, 'ymax': 62}},
    {'score': 0.5866526961326599, 'label': 'LABEL_1', 'box': {'xmin': 307, 'ymin': 109, 'xmax': 414, 'ymax': 231}}
]
"""
print(f"Prediction time: {end_prediction_time-start_prediction_time} seconds")

res = np.array(res)
img = np.array(img)
img = img[:,:,::-1].copy()
start_draw_time = time()
img = draw_boxes(img,res)
end_draw_time = time()
print(f"draw time: {end_draw_time-start_draw_time}")

cv2.imshow("img",img)
cv2.waitKey(0)


class TreeTopViewDetector_YOLOS():
    def __init__(self,model_dir,threshold=0.5):
        self.FPS = 0
        self.model = self.load_model(model_dir,threshold)
    
    def load_model(model_dir,threshold=0.5):
        pipe = pipeline("object-detection", model=model_dir, threshold=threshold)
        return pipe
    
    def get_label(self,classes,box_data):
        label = box_data["label"]
        index = label.split("_")[-1]
        return classes[int(index)-1]
    
    def draw_box(img,box_data,color=(0,255,00),thickness=2,FPS=0):
        xmin = box_data["box"]["xmin"]
        ymin = box_data["box"]["ymin"]
        xmax = box_data["box"]["xmax"]
        ymax = box_data["box"]["ymax"]
        label = get_label(class_names,box_data)
        score = box_data["score"]
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color,thickness)
        cv2.putText(img,f"{label}({score:0.3f})",(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        cv2.putText(img,f"{FPS:0.3f} FPS",(xmin,ymin+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,100,255),1)
        return img
    
    def draw_boxes(img,box_data,color=(0,255,00),thickness=2):
        for box in box_data:
            img = draw_box(img,box,color,thickness)
        return img
    
    def predict(self,img):
        start_time = time()
        self.img = img
        self.prediction = self.model(img)
        self.FPS = 1/(time()-start_time)
        return self.prediction
    
    def draw(self,img=None,color=(0,255,00),thickness=2):
        if img is not None:
            self.img = img
        self.img = self.img[:,:,::-1].copy()
        self.img = self.draw_boxes(self.img,self.prediction,color,thickness)
        return self.img