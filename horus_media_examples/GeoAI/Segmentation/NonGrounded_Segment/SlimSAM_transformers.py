import os
os.environ['TRANSFORMERS_CACHE'] = 'horus_media_examples/GeoAI/models/'
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor

DEVICE = 'cuda:0'
def show_mask(masks, ax):
    for mask in masks:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

def save_masks_on_image(pil_image, masks, outputFileName):
    plt.figure(figsize=(10, 10),frameon=False)
    plt.imshow(pil_image)
    for mask in masks: show_mask(mask.cpu().numpy(), plt.gca())
    plt.axis('off')
    plt.savefig(outputFileName,bbox_inches='tight', pad_inches = 0)

model = SamModel.from_pretrained("nielsr/slimsam-77-uniform").to(DEVICE).eval()            # nielsr/slimsam-50-uniform of nielsr/slimsam-77-uniform
processor = SamProcessor.from_pretrained("nielsr/slimsam-77-uniform")


def processSlimSam(pil_image, bboxes, multiMask):
    inputs = processor(pil_image, input_boxes=[bboxes], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=multiMask)       # , multimask_output=False   <<== Let's just output a single mask per box
    return inputs, outputs

def processSlimSam_All(pil_image, bboxes, outputFileName, multiMask=False):
    inputs, outputs = processSlimSam(pil_image=pil_image,bboxes=bboxes,multiMask=multiMask)
    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    save_masks_on_image(pil_image=pil_image,masks=masks,outputFileName=outputFileName)


if __name__ == "__main__":
    pil_image = Image.open('horus_media_examples/GeoAI/input/test.png')
    #Input box [[[x1,y1,x2,y2]]]
    input_boxes = [[[917, 305, 958, 581],[842, 455, 883, 642],[1198, 414, 1219, 538],[560, 390, 574, 562]]]
    processSlimSam_All(pil_image=pil_image,bboxes=input_boxes, outputFileName='horus_media_examples/GeoAI/output/test.png', multiMask=False)


