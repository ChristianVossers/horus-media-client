
import os
os.environ['TRANSFORMERS_CACHE'] = 'horus_media_examples/GeoAI/models/'

from PIL import Image
import numpy as np
import torch
import cv2

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from transformers import AutoProcessor

device = "cuda:0" #"cpu"

labels_of_interest_cityscapes = [5,6,7]
labels_of_interest_mapillary  = [44,45,46,47,48,49,50]

# feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
# model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to(device)
# feature_extractor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
# model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic").to(device)
# feature_extractor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic")
# model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic").to(device)
feature_extractor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_cityscapes_swin_large").to(device)

model.eval()
print ('MODEL LOADED')

def processTransformers(image, labels_of_interest):
    inputs = feature_extractor(images=image, task_inputs=["semantic"], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_segmentation_map = feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0].cpu().numpy()

    seg_mask = np.uint8(predicted_segmentation_map)
    seg_mask[~np.any(seg_mask == np.array(labels_of_interest)[:, None, None], axis = 0)] = 0
    seg_mask[seg_mask > 0] = 1
    _, labels = cv2.connectedComponents(seg_mask)
    return seg_mask, labels

def processTransformers_All(image, labels_of_interest, outputfilename):
    seg_mask, labels = processTransformers(image, labels_of_interest)
    cv2.imwrite(outputfilename + '_masks.png', labels)
    img = image
    seg_img = Image.fromarray(seg_mask*255)
    img.paste(seg_img, (0, 0), seg_img)
    img.save(outputfilename)

# image = Image.open('horus_media_examples/GeoAI/input/img_01120_1_ab7f2e61-a1ba-4c61-81ae-81a136cc54bb.png')
# processTransformers_All(image, labels_of_interest_cityscapes, "_ShiLabs-oneformer-large-cityscapes.png")
