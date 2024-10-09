from ultralytics import SAM
import cv2

model = SAM('horus_media_examples/GeoAI/models/sam2_l.pt') # sam2_b.pt

def processSAM_V2_Ultralytics(cv2_image, bboxes, ModelResolution):
    return model.predict(cv2_image, bboxes=bboxes, device='cuda', verbose=False, imgsz=ModelResolution)  #

def processSAM_V2_Ultralytics_All(cv2_image, bboxes, outputFileName, modelResolution=1024):
    results = processSAM_V2_Ultralytics(cv2_image=cv2_image,bboxes=bboxes,ModelResolution=modelResolution)
    for result in results:
        result.save(outputFileName, conf=True, boxes=True, line_width=1)

if __name__ == "__main__":
    cv2_image = cv2.imread('horus_media_examples/GeoAI/input/test.png')
    bboxes = [[917, 305, 958, 581],[842, 455, 883, 642],[1198, 414, 1219, 538],[560, 390, 574, 562]]
    processSAM_V2_Ultralytics_All(cv2_image=cv2_image,bboxes=bboxes, outputFileName='horus_media_examples/GeoAI/output/test.png', modelResolution=1920)
