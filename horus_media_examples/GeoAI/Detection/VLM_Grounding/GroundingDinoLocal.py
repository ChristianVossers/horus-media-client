from groundingdino.util.inference import load_model, predict, annotate
import groundingdino.datasets.transforms as T
from PIL import Image
# import os
# import numpy as np

model = load_model('horus_media_examples/GeoAI/config/groundingdino_Base.py', 'horus_media_examples/GeoAI/models/groundingdino_Base.geoAI',device="cuda:0")
#model = load_model('horus_media_examples/GeoAI/config/groundingdino_Tiny.py', 'horus_media_examples/GeoAI/models/groundingdino_Tiny.geoAI',device="cuda:0")
#model = load_model('horus_media_examples/GeoAI/config/config_cfg_gdinot-1.8m-odvg.py', 'horus_media_examples/GeoAI/models/gdinot-1.8m-odvg.pth',device="cuda:0")

def load_image(pil_image):
    transform = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image_transformed, _ = transform(pil_image, None)
    return image_transformed


def processGroundingDino(pil_image, text="pole", box_threshold=0.2, text_threshold=0.2):
    image = load_image(pil_image)
    return predict(model=model, image=image, caption=text, box_threshold=box_threshold, text_threshold=text_threshold,device="cuda:0")


def processGroundingDino_All(pil_image, np_image, outputFileName, text="pole", box_threshold=0.18, text_threshold=0.18):
    boxes, logits, phrases = processGroundingDino(pil_image=pil_image,text=text, box_threshold=box_threshold, text_threshold=text_threshold)
    annotated_frame = annotate(image_source=np_image, boxes=boxes, logits=logits, phrases=phrases)
    pil_annotated_frame = Image.fromarray(annotated_frame)
    pil_annotated_frame.save(outputFileName)


#-----------------OneDefaultImage------------------
# pil_image = Image.open('horus_media_examples/GeoAI/input/img_01120_1_ab7f2e61-a1ba-4c61-81ae-81a136cc54bb.png')
# np_image = np.asarray(pil_image)[:, :, ::-1]        
# processGroundingDino_All(pil_image,text="pole", np_image=np_image,outputFileName='horus_media_examples/output/test.png', box_threshold=0.18,text_threshold=0.18)

#-----------------Dataset1724------------------
# dir = "C:/Horus/Horus Input en Data/data_scale1_shortclip/"
# fullName = []
# fileNames = []
# for root, dirs, files in os.walk(os.path.abspath(dir)):
#     for file in files:
#         fullName.append(os.path.join(root, file))
#         fileNames.append(file)
# for count in range (0,len(fileNames)):
#     pil_image = Image.open(fullName[count])
#     np_image = np.asarray(pil_image)[:, :, ::-1]        
#     processGroundingDino_All(pil_image,text="pole", np_image=np_image,box_threshold=0.18,text_threshold=0.18)


