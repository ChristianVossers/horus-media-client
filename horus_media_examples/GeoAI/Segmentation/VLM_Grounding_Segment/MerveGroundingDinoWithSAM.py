import os
os.environ['TRANSFORMERS_CACHE'] = 'horus_media_examples/GeoAI/models/'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from transformers import SamModel, SamProcessor
import numpy as np
from PIL import Image


text = ["pole"]
box_threshold=0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# sam_model = SamModel.from_pretrained("facebook/sam-vit-large").to(device)
# sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")


# dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(device)
# dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")

def processMerveDinoGrounding(img, text, box_threshold):
  textstr=""
  for query in text:
    textstr += f"{query}. "
  width, height = img.shape[:2]
  target_sizes=[(width, height)]
  inputs = dino_processor(text=textstr, images=img, return_tensors="pt").to(device)
  with torch.no_grad():
    outputs = dino_model(**inputs)
    #outputs.logits = outputs.logits.cpu()
    #outputs.pred_boxes = outputs.pred_boxes.cpu()
    results = dino_processor.post_process_grounded_object_detection(outputs=outputs, input_ids=inputs.input_ids,box_threshold=box_threshold,target_sizes=target_sizes)
  return results

def processMerveDinoGroundedSam(np_image, text, box_threshold):
  dino_output = processMerveDinoGrounding(np_image, text, box_threshold)
  result_labels=[]
  for pred in dino_output:
    boxes = pred["boxes"].cpu()
    scores = pred["scores"].cpu()
    labels = pred["labels"]
    box = [torch.round(pred["boxes"][0], decimals=2), torch.round(pred["boxes"][1], decimals=2), torch.round(pred["boxes"][2], decimals=2), torch.round(pred["boxes"][3], decimals=2)]
    for box, score, label in zip(boxes, scores, labels):
      if label != "":
        inputs = sam_processor(np_image,input_boxes=[[[box]]],return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = sam_model(**inputs)
        mask = sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(),inputs["original_sizes"].cpu(),inputs["reshaped_input_sizes"].cpu())[0][0][0].numpy()
        mask = mask[np.newaxis, ...]
        result_labels.append((mask, label))
  return np_image, result_labels

def processMerveDinoGroundedSam_All(pil_image, np_image, text, outputFileName, box_threshold=box_threshold):
  outimg, result_labels = processMerveDinoGroundedSam(np_image,text,box_threshold)
  img_masks = Image.new(mode='RGB', size=(outimg.shape[1],outimg.shape[0]))
  mask_count = 1
  for result_label in result_labels:
      mask = np.squeeze(result_label[0])
      maskpil = Image.fromarray(mask)
      mask = Image.fromarray((mask.astype(int) * mask_count).astype(np.uint8))
      img_masks.paste(mask, (0,0), maskpil)
      pil_image.paste(maskpil, (0,0), maskpil)
      mask_count += 1
  #img_masks.save('./output/' + fileNames[count] + '_masks.png')
  pil_image.save(outputFileName)



######################### Detect met Transformers grounding-dino-base, Segmentatie met Transformers SAM-VIT-Base ########################
# fullName = []
# fileNames = []
# for root, dirs, files in os.walk(os.path.abspath("C:/Horus/Horus Input en Data/data_scale1_shortclip/")):
#     for file in files:
#         fullName.append(os.path.join(root, file))
#         fileNames.append(file)
# for count in range (0,len(fileNames)):
#     pil_image = Image.open(fullName[count])
#     np_image = np.asarray(pil_image)[:, :, ::-1]        
#     processMerveDinoGroundedSam_All(pil_image=pil_image,np_image=np_image,text=text,box_threshold=box_threshold, outputFileName='horus_media_examples/GeoAI/output/' + str(count) + '.png')
    
    
