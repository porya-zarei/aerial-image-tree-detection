from PIL import Image
import cv2
from TreeTopViewDetector_YOLOv8 import TreeTopViewDetector_YOLOv8
img_paths = [
    "../../data/xml/test/ABBY_027_2019_jpeg.rf.3bd16408d12e7f8e1ceaa3b44a8a8441.jpg",
    "../../data/xml/test/2018_SJER_3_252000_4108000_image_517_jpeg.rf.2140d7254f469763e42c1f9d32ec0dac.jpg",
    "../../data/xml/test/BART_071_2019_jpeg.rf.79d786245f7126a6a059e498287f404a.jpg",
    "../../data/xml/test/JERC_054_2019_jpeg.rf.6b08fe7b77c0e76db72271875441b10f.jpg",
    "../../data/xml/test/DEJU_021_2019_jpeg.rf.4836d61e1193cf782be67de9c3c5a4d6.jpg",
    "../../data/xml/test/HEAL_022_2019_jpeg.rf.5c7e203ab6e5eaeb18ecd661a7fbf55d.jpg",
    "../../data/xml/test/HEAL_020_2018_jpeg.rf.245d6c7663455e7a946c0f5ab6245a35.jpg"
]
imgs = [Image.open(i).convert("RGB") for i in img_paths]

# video = cv2.VideoCapture("../../../top-tree-detection/src/data/testing/video.mp4")

class_names = ["tree-top"]
# Step 1: Initialize model with the best available weights
# YOLOS model
weights_path = "./best.pt"

predictor = TreeTopViewDetector_YOLOv8(weights_path,class_names,threshold=0.3)

for i in range(len(imgs)):
    img = imgs[i]
    predictor.predict(img)
    img = predictor.draw(img)
    cv2.imshow(f"img-{i}",img)

# while True:
#     ret, frame = video.read()
#     if not ret:
#         break
#     img = Image.fromarray(frame)
#     predictor.predict(img)
#     img = predictor.draw(img)
#     cv2.imshow("img",img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

cv2.waitKey(0)
