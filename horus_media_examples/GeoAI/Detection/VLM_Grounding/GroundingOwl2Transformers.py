import os
os.environ['TRANSFORMERS_CACHE'] = 'horus_media_examples/GeoAI/models/'
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image, ImageDraw
import torch

# Veel -> Weinig detectie (al dan niet juist)
#google/owlv2-base-patch16
#google/owlv2-base-patch16-finetuned
#google/owlv2-large-patch14-finetuned
#google/owlv2-large-patch14-ensemble

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-finetuned", device="cuda")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-finetuned").to("cuda")

def saveResultImage(pil_image, outputFilename, texts, boxes, scores, labels):
    draw = ImageDraw.Draw(pil_image)
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        x1, y1, x2, y2 = tuple(box)
        draw.rectangle(xy=((x1, y1), (x2, y2)), outline="red")
        draw.text(xy=(x1, y1), text=texts[0][label] + str(round(score.item(), 3)))
    pil_image.save(outputFilename)

def processOwl2(pil_image, text, box_threshold):
    inputs = processor(images=pil_image, text=[text], return_tensors="pt")
    inputs.to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=torch.Tensor([pil_image.size[::-1]]), threshold=box_threshold)
    return results[0]["boxes"], results[0]["scores"], results[0]["labels"]

def processOwl2_All(pil_image, outputFilename, text, box_threshold=0.21):
    boxes, scores, labels = processOwl2(pil_image=pil_image,text=text,box_threshold=box_threshold)
    saveResultImage(pil_image=pil_image, outputFilename=outputFilename,texts=text,boxes=boxes,scores=scores,labels=labels)

# pil_image = Image.open('horus_media_examples/GeoAI/input/img_01120_1_ab7f2e61-a1ba-4c61-81ae-81a136cc54bb.png')
# texts = [['pole', 'traffic light']]
# processOwl2_All(pil_image=pil_image,outputFilename='horus_media_examples/GeoAI/output/test.png', text=texts,box_threshold=0.21)



