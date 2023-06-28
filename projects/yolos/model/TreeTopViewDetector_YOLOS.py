import numpy as np
from transformers import pipeline
import cv2
from time import time

class TreeTopViewDetector_YOLOS():
    def __init__(self,model_dir,class_names=[],threshold=0.5):
        self.FPS = 0
        self.class_names = class_names
        self.model = self.__load_model(model_dir,threshold)
    
    def __load_model(self,model_dir,threshold=0.5):
        pipe = pipeline("object-detection", model=model_dir, threshold=threshold)
        pipe.model.eval()
        return pipe
    
    def __get_label(self,classes,box_data):
        label = box_data["label"]
        index = label.split("_")[-1]
        return classes[int(index)-1]
    
    def __draw_box(self,img,box_data,color=(0,255,00),thickness=2):
        xmin = box_data["box"]["xmin"]
        ymin = box_data["box"]["ymin"]
        xmax = box_data["box"]["xmax"]
        ymax = box_data["box"]["ymax"]
        label = self.__get_label(self.class_names,box_data)
        score = box_data["score"]
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color,thickness)
        cv2.putText(img,f"{label}({score:0.3f})",(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        return img
    
    def __draw_boxes(self,img,box_data,color=(0,255,00),thickness=2):
        for box in box_data:
            img = self.__draw_box(img,box,color,thickness)
        cv2.putText(img,f"{self.FPS:0.3f} FPS",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(200,0,0),1)
        return img
    
    def predict(self,img):
        """
        [
            {'score': 0.13099995255470276, 'label': 'LABEL_1', 'box': {'xmin': 83, 'ymin': 151, 'xmax': 165, 'ymax': 246}}, # box_data
            {'score': 0.8488465547561646, 'label': 'LABEL_1', 'box': {'xmin': 315, 'ymin': 138, 'xmax': 409, 'ymax': 228}},
            {'score': 0.40752533078193665, 'label': 'LABEL_1', 'box': {'xmin': 43, 'ymin': 26, 'xmax': 74, 'ymax': 62}},
            {'score': 0.5866526961326599, 'label': 'LABEL_1', 'box': {'xmin': 307, 'ymin': 109, 'xmax': 414, 'ymax': 231}}
        ]
        """
        start_time = time()
        self.img = img
        self.prediction = self.model(self.img)
        self.FPS = 1.0/(time()-start_time)
        return self.prediction
    
    def draw(self,img=None,color=(0,255,00),thickness=2):
        if img is not None:
            self.img = img
        self.img = np.array(img)
        self.img = self.img[:,:,::-1].copy()
        self.img = self.__draw_boxes(self.img,self.prediction,color,thickness)
        return self.img