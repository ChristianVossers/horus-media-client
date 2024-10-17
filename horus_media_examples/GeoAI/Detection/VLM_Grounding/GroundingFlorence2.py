import os
os.environ['TRANSFORMERS_CACHE'] = 'horus_media_examples/GeoAI/models/'
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch

colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red','lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

# 'microsoft/Florence-2-base',
# 'microsoft/Florence-2-base-ft',
# 'microsoft/Florence-2-large',
# 'microsoft/Florence-2-large-ft',  <== STD
# 'thwri/CogFlorence-2.1-Large',
# 'thwri/CogFlorence-2.2-Large',
# 'gokaygokay/Florence-2-Flux-Large',  <=='Anders'

# "precision": 'fp16','bf16','fp32'
# "attention": 'flash_attention_2', 'sdpa', 'eager'
dtype = torch.float16 #"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32
model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large-ft', attn_implementation='sdpa', device_map="cuda", torch_dtype=dtype,trust_remote_code=True).eval().cuda()
processor = AutoProcessor.from_pretrained('microsoft/Florence-2-large-ft', trust_remote_code=True)

CmdGROUNDING = '<CAPTION_TO_PHRASE_GROUNDING>'

def save_bbox_image(image, data, outputFilename):
    figsize = (1920,1080)
    plt.figure(figsize=figsize,dpi=96)
    _, ax = plt.subplots()                                                          # Create a figure and axes
    ax.imshow(image)                                                                # Display the image
    for bbox, label in zip(data['bboxes'], data['labels']):                         # Plot each bounding box
        x1, y1, x2, y2 = bbox                                                       # Unpack the bounding box coordinates
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)                                                          # Add the rectangle to the Axes
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))       # Annotate the label
    ax.axis('off')                                                                  # Remove the axis ticks and labels
    plt.savefig(outputFilename, bbox_inches = 'tight',pad_inches = 0)

def processFlorence2(pil_image, text, command):
    prompt = command + " " + text
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt", do_rescale=False).to('cuda', dtype)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=9,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text,task=command,image_size=(pil_image.width, pil_image.height))
    return parsed_answer

def processFlorence2Grounding_All(pil_image, outputFilename, text='street light'):
    results = processFlorence2(pil_image=pil_image, text=text,command=CmdGROUNDING)
    save_bbox_image(pil_image, results[CmdGROUNDING], outputFilename)

# pil_image = Image.open('horus_media_examples/GeoAI/input/img_01120_1_ab7f2e61-a1ba-4c61-81ae-81a136cc54bb.png')
# processFlorence2Grounding_All(pil_image, text="street light", outputfilename='horus_media_examples/GeoAI/output/test.png')





