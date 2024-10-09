import os
from PIL import Image
import supervision as sv
from inference.models import YOLOWorld

model = YOLOWorld(model_id="yolo_world/l")

classes = ["pole"]  #, "mast", "traffic sign"
model.set_classes(classes)

def saveResultsToImage(image, detections, outputFilename):
    BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
    LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=1, text_scale=0.3, text_color=sv.Color.BLACK)
    labels = [f"{classes[class_id]} {confidence:0.7f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
    annotated_image = image.copy()
    annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
    annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections, labels=labels)
    annotated_image.save(outputFilename)

def processYoloWorld(pil_image, box_threshold, iou_threshold, nms_threshold):
    results = model.infer(pil_image, confidence=box_threshold,iou_threshold=iou_threshold)
    detections = sv.Detections.from_inference(results)#.with_nms(threshold=nms_threshold)
    return detections

def processYoloWorld_All(pil_image, outputFileName, box_threshold=0.00025, iou_threshold=0.3, nms_threshold=0.1):       #Box = 0.00025...0.00052
    detections = processYoloWorld(pil_image=pil_image,box_threshold=box_threshold,iou_threshold=iou_threshold, nms_threshold=0.1)
    saveResultsToImage(image=pil_image,detections=detections,outputFilename=outputFileName)


#-----------------Dataset1724------------------
# dir = "C:/Horus/Horus Input en Data/data_scale1_shortclip_1024"
# fullName = []
# fileNames = []
# for root, dirs, files in os.walk(os.path.abspath(dir)):
#     for file in files:
#         fullName.append(os.path.join(root, file))
#         fileNames.append(file)

# for count in range (0,len(fileNames)):
#     pil_image = Image.open(fullName[count])
#     processYoloWorld_All(pil_image=pil_image,outputFileName='horus_media_examples/GeoAI/output/' + str(count) + '.png')

    





#-----------------EXAMPLE------------------
# results = model.infer(image, confidence=0.00001,iou_threshold=0.00001)
# detections = sv.Detections.from_inference(results)#.with_nms(threshold=0.1)
# BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
# LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)
# labels = [f"{classes[class_id]} {confidence:0.4f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
# annotated_image = image.copy()
# annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
# annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections, labels=labels)
# cv2.imwrite('output.png', annotated_image)
