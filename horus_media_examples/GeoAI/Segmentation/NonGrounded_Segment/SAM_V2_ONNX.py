from .sam2_onnx import SAM2Image, draw_masks
import cv2
import numpy as np

sam2 = SAM2Image("horus_media_examples/GeoAI/models/sam2_base_plus_encoder.geoAI", 
                 "horus_media_examples/GeoAI/models/sam2_base_decoder.geoAI")                         # Initialize models

def processSAM_V2_onnx(np_image, bboxes):
    sam2.set_image(np_image)                                                                                    # Set image
    for count, coords in enumerate(bboxes):                                                                     # Add points
        sam2.set_box(((coords[0],coords[1]),(coords[2],coords[3])),label_id=count)
    return sam2.get_masks()

def processSAM_V2_onnx_All(np_image, bboxes, outputFileName):
    masks = processSAM_V2_onnx(np_image=np_image,bboxes=bboxes)
    masked_img = draw_masks(np_image, masks)                                                                    # Draw masks on original
    cv2.imwrite(outputFileName,masked_img)
    # cv2.imshow("masked_img", masked_img)
    # cv2.waitKey(0)

if __name__ == "__main__":
    np_image = np.array(cv2.imread("horus_media_examples/GeoAI/input/test.png"))
    point_coords = [(566, 458), (861,487), (941, 417), (1208, 445),(1291, 366)]                                     
    bboxes = [[917, 305, 958, 581],[842, 455, 883, 642],[1198, 414, 1219, 538],[560, 390, 574, 562]]
    processSAM_V2_onnx_All(np_image=np_image,bboxes=bboxes,outputFileName='horus_media_examples/GeoAI/output/test.png')