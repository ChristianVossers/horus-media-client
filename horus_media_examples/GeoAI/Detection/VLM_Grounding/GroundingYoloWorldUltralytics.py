import os
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLOWorld
import random
from GeoAI.BBoxResult import BBoxResult

DEVICE = 'cuda:0'
RESOLUTION = (672,2048)

CONFIDENCE = 0.001                                                  # Minimale zekerheid in tracking om track vast te houden
CLASS_IOU = 0.1
AUGMENT = False                                                     # Augment betekend: extra tries in detectie. Meer robuust maar trager   (Los geen effect op detectie)
AGNOSTIC_NMS = False                                                # Overlappende track detecties samenvoegen                              (Los geen effect op detectie)
RETINA_MASKS = False                                                # Extra hoge resolutie in segmentatie                                  (Los geen effect op detectie)

def generate_colors(num_colors):
    return [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(num_colors)]

model = YOLOWorld("horus_media_examples/GeoAI/models/yolov8l-worldv2.pt")                                         # Load a pretrained YOLOv8s-worldv2 model
classes = ["pole"]
model.set_classes(classes)
class_colors = {cls: color for cls, color in zip(classes, generate_colors(len(classes)))}

def annotate_image(image, result):
    for box in result.boxes:                                                                # Extract bounding boxes from the results
        x1, y1, x2, y2 = map(int, box.xyxy[0])                                                  # Get bounding box coordinates
        conf = box.conf[0]                                                                      # Confidence score of the detection
        cls_id = int(box.cls[0])                                                                # Class ID of the detected object
        label = f"{conf:.4f}"                                        # Label with class name and confidence
        color = class_colors[result.names[cls_id]]                                          # Color for the class
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)                                      # Draw the bounding box
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)                    # Calculate the text size and position
        cv2.rectangle(image, (x1, y1), (x1 + w, y1+h+3), color, -1)                        # Background rectangle for label
        cv2.putText(image, label, (x1, y1 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)    # Text label in black color
    return image

# def get_bboxes(result):
#     bboxes = []
#     for box in result.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         bboxes.append(BBoxResult(x1=x1,y1=y1,x2=x2,y2=y2,conf=float(box.conf[0]),classID=int(box.cls[0]) + 1))
#     return bboxes

# def processYoloWorld_toBBox(pil_image, box_threshold=CONFIDENCE, iou_threshold=CLASS_IOU):
#     return get_bboxes(model.predict(source=pil_image, show=False, save=False, verbose=False,stream=False,show_conf=True,
#         device=DEVICE, imgsz=RESOLUTION, conf=box_threshold, iou=iou_threshold, augment=AUGMENT,agnostic_nms=AGNOSTIC_NMS,retina_masks=RETINA_MASKS)[0])

def processYoloWorld(pil_image, box_threshold=CONFIDENCE, iou_threshold=CLASS_IOU):
    return model.predict(source=pil_image, show=False, save=False, verbose=False,stream=False,show_conf=True,
        device=DEVICE, imgsz=RESOLUTION, conf=box_threshold, iou=iou_threshold, augment=AUGMENT,agnostic_nms=AGNOSTIC_NMS,retina_masks=RETINA_MASKS)[0]

def processYoloWorld_All(pil_image, outputFileName, box_threshold=CONFIDENCE, iou_threshold=CLASS_IOU):
    results = processYoloWorld(pil_image=pil_image, box_threshold=box_threshold, iou_threshold=iou_threshold)
    annotated_frame = annotate_image(pil_image, results)
    cv2.imwrite(outputFileName, annotated_frame)


#-----------------Dataset1724------------------
# dir = "C:/Horus/Horus Input en Data/data_scale1_shortclip/"
# fullName = []
# fileNames = []
# classes = ["pole"]
# class_colors = {cls: color for cls, color in zip(classes, generate_colors(len(classes)))}

# for root, dirs, files in os.walk(os.path.abspath(dir)):
#     for file in files:
#         fullName.append(os.path.join(root, file))
#         fileNames.append(file)

# for count in range (0,len(fileNames)):
#     pil_image = Image.open(fullName[count])
#     np_image = np.array(pil_image)
#     cv_image = cv2.imread(fullName[count])
#     processYoloWorld_All(image=pil_image, text=classes, outputFileName='horus_media_examples/GeoAI/output/' + str(count) + '.png', box_threshold=CONFIDENCE, iou_threshold=CLASS_IOU)
    
    

    









