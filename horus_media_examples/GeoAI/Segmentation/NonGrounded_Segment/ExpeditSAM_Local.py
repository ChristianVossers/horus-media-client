import numpy as np
import torch
import cv2
from .segment_anything_expedit import SamPredictor,sam_model_registry
import matplotlib.pyplot as plt

DEVICE = 'cuda:0'
sam = sam_model_registry["vit_h"](checkpoint='horus_media_examples/GeoAI/models/sam_V1_Huge.geoAI', use_hourglass=True).to(device=DEVICE)
sam = SamPredictor(sam)

def show_mask(mask, ax):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def processExpeditSAM(cv2_image, bboxes, multiMask):
    sam.set_image(cv2_image)
    transformed_boxes = sam.transform.apply_boxes_torch(bboxes, cv2_image.shape[:2])
    return sam.predict_torch(point_coords=None,point_labels=None,boxes=transformed_boxes,multimask_output=multiMask)

def processExpeditSAM_All(cv2_image, bboxes, outputFileName, multiMask=False):
    bboxes = torch.tensor(bboxes, device=DEVICE)
    masks, _, _ = processExpeditSAM(cv2_image=cv2_image,bboxes=bboxes,multiMask=multiMask)

    plt.figure(figsize=(10, 10),frameon=False)
    plt.imshow(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    for mask in masks: show_mask(mask.cpu().numpy(), plt.gca())
    for box in bboxes: show_box(box.cpu().numpy(), plt.gca())
    plt.axis('off')
    plt.savefig(outputFileName,bbox_inches='tight', pad_inches = 0)


if __name__ == "__main__":
    cv2_image = cv2.imread('horus_media_examples/GeoAI/input/test.png')
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    bboxes = [
        [917, 305, 958, 581],
        [842, 455, 883, 642],
        [1198, 414, 1219, 538],
        [560, 390, 574, 562],
    ]
    processExpeditSAM_All(cv2_image=cv2_image,bboxes=bboxes, outputFileName='horus_media_examples/GeoAI/output/test.png', multiMask=False)











