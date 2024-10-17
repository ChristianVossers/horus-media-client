import os
os.environ['TRANSFORMERS_CACHE'] = 'horus_media_examples/GeoAI/models/'
from PIL import Image
from transformers import pipeline
import matplotlib.pyplot as plt
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],[0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

pipe = pipeline(task="zero-shot-object-detection", model="IDEA-Research/grounding-dino-base",device="cuda")

def save_results(pil_img, scores, labels, boxes, outputFileName):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,fill=False, color=c, linewidth=3))
        label = f'{"pole"}: {score:0.2f}'
        ax.text(xmin, ymin, label, fontsize=15,bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig(outputFileName,bbox_inches='tight',pad_inches = 0)
    
def processGroundingDinoPipeline(pil_image, text, box_threshold):
    results = pipe(pil_image, candidate_labels=text, threshold=box_threshold)
    scores, labels, boxes = [], [], []
    for result in results:
        scores.append(result["score"])
        labels.append(result["label"])
        boxes.append(tuple(result["box"].values()))
    return scores, labels, boxes

def processGroundingDinoPipeline_All(pil_image, outputFileName, text=["pole."], box_threshold=0.25):
    scores, labels, boxes = processGroundingDinoPipeline(pil_image, text=text, box_threshold=box_threshold)
    save_results(pil_image, scores, labels, boxes, outputFileName=outputFileName)


############################# One image ##############################
pil_image = Image.open('horus_media_examples/GeoAI/input/img_01120_1_ab7f2e61-a1ba-4c61-81ae-81a136cc54bb.png')
processGroundingDinoPipeline_All(pil_image=pil_image,text=["pole."],outputFileName='horus_media_examples/GeoAI/output/test.png')

