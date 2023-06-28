import cv2
import torch
import supervision as sv
from transformers import DetrForObjectDetection, DetrImageProcessor,DeformableDetrForObjectDetection,DeformableDetrImageProcessor
from time import time


# settings
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# CHECKPOINT = 'facebook/detr-resnet-50'
CHECKPOINT = './'
CONFIDENCE_TRESHOLD = 0.5
IOU_TRESHOLD = 0.8
IMAGE_PATH = "../../data/org/images/test/HEAL_022_2018.jpeg"

image_processor = DetrImageProcessor.from_pretrained("./config.json")
model = DetrForObjectDetection.from_pretrained(CHECKPOINT)
# image_processor = DeformableDetrImageProcessor.from_pretrained("./config.json")
# model = DeformableDetrForObjectDetection.from_pretrained(CHECKPOINT,ignore_mismatched_sizes=True)
model.to(DEVICE)

with torch.no_grad():
    # load image and predict
    image = cv2.imread(IMAGE_PATH)
    start_time = time()
    inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
    outputs = model(**inputs)
    end_time = time()
    print(f"time => {end_time - start_time}")
    # post-process
    target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
    results = image_processor.post_process_object_detection(
        outputs=outputs, 
        threshold=CONFIDENCE_TRESHOLD, 
        target_sizes=target_sizes
    )[0]

# annotate
detections = sv.Detections.from_transformers(transformers_results=results)
new_labels = []
new_detections = []
wanted_class = "dog"

for data in detections:
    _1, _2,confidence, class_id, _3 = data
    cl_name = model.config.id2label[class_id]
    if True: # cl_name.count(wanted_class) > 0
        new_labels.append(f"{cl_name} {confidence:0.2f}")
        new_detections.append([ _1, confidence, class_id, _2])
box_annotator = sv.BoxAnnotator()
frame = box_annotator.annotate(scene=image, detections=detections, labels=new_labels,skip_label=True)

sv.plot_image(frame, (16, 16))