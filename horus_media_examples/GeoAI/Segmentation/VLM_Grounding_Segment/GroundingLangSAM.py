import os
import numpy as np
from PIL import Image
from samgeo.text_sam import LangSAM

sam = LangSAM()

def processGroundedLangSam(pil_image, text, box_threshold,text_threshold):
    masks, boxes, phrases, logits = sam.predict(pil_image, text_prompt=text, box_threshold=box_threshold, text_threshold=text_threshold, return_results=True)
    return masks, boxes, phrases, logits

def processGroundedLangSam_All(pil_image, text, outputFileName, box_threshold=0.24, text_threshold=0.2):
    masks, boxes, phrases, logits = processGroundedLangSam(pil_image, text, box_threshold,text_threshold)
    #sam.show_anns(cmap="Greys_r",add_boxes=False,alpha=1,title="",blend=True,output=outputFileName + "_mask.png")
    sam.show_anns(cmap="Greens", add_boxes=False,alpha=0.5,title=text,blend=True,output=outputFileName)


######################### Detect met Transformers grounding-dino-base, Segmentatie met Transformers SAM-VIT-Base ########################
# fullName = []
# fileNames = []
# for root, dirs, files in os.walk(os.path.abspath("C:/Horus/Horus Input en Data/data_scale1_shortclip/")):
#     for file in files:
#         fullName.append(os.path.join(root, file))
#         fileNames.append(file)
# for count in range (0,len(fileNames)):
#     pil_image = Image.open(fullName[count])   
#     processGroundedLangSam_All(pil_image=pil_image,text="pole",box_threshold=0.24,text_threshold=0.24, outputFileName='horus_media_examples/GeoAI/output/' + str(count) + '.png')
    
    


