import os
os.environ['TRANSFORMERS_CACHE'] = 'horus_media_examples/GeoAI/models/'
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw
import random
import numpy as np
import torch

colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red','lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

# 'microsoft/Florence-2-base',
# 'microsoft/Florence-2-base-ft',
# 'microsoft/Florence-2-large',
# 'microsoft/Florence-2-large-ft',
# 'HuggingFaceM4/Florence-2-DocVQA',
# 'thwri/CogFlorence-2.1-Large',
# 'thwri/CogFlorence-2.2-Large',
# 'gokaygokay/Florence-2-SD3-Captioner',
# 'gokaygokay/Florence-2-Flux-Large',
# 'MiaoshouAI/Florence-2-base-PromptGen-v1.5',
# 'MiaoshouAI/Florence-2-large-PromptGen-v1.5'

# "precision": 'fp16','bf16','fp32'
# "attention": 'flash_attention_2', 'sdpa', 'eager'
dtype = torch.float16 #"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32
model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large-ft', attn_implementation='sdpa', device_map="cuda", torch_dtype=dtype,trust_remote_code=True).eval().cuda()
processor = AutoProcessor.from_pretrained('microsoft/Florence-2-large-ft', trust_remote_code=True)

CmdGROUNDING = '<CAPTION_TO_PHRASE_GROUNDING>'
CmdSEGMENT   = '<REFERRING_EXPRESSION_SEGMENTATION>'

def save_polygon_image(pil_image, prediction, outputfilename, fill_mask=False):
    draw = ImageDraw.Draw(pil_image)                                                    # Load the image
    scale = 1                                                                       # Set up scale factor if needed (use 1 if not scaling)
    for polygons, label in zip(prediction['polygons'], prediction['labels']):       # Iterate over polygons and labels
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue
            _polygon = (_polygon * scale).reshape(-1).tolist()
            if fill_mask: draw.polygon(_polygon, outline=color, fill=fill_color)    # Draw the polygon
            else: draw.polygon(_polygon, outline=color)
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)        # Draw the label text
    pil_image.save(outputfilename)

def processFlorence2(pil_image, text, command):
    prompt = command + " " + text
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt", do_rescale=False).to('cuda', dtype)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text,task=command,image_size=(pil_image.width, pil_image.height))
    return parsed_answer

def processFlorence2Segment_All(pil_image, outputFilename, text='street light'):
    results = processFlorence2(pil_image=pil_image, text=text,command=CmdSEGMENT)
    save_polygon_image(pil_image, results[CmdSEGMENT], outputFilename, fill_mask=True)


# pil_image = Image.open('horus_media_examples/GeoAI/input/img_01120_1_ab7f2e61-a1ba-4c61-81ae-81a136cc54bb.png')
# processFlorence2Segment_All(pil_image,text="street light",outputFilename='horus_media_examples/GeoAI/output/test.png')
