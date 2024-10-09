
from mmseg.apis import MMSegInferencer
import torch
torch.hub.set_dir("horus_media_examples/GeoAI/models")
from PIL import Image
import numpy as np
import cv2

labels_of_interest_cityscapes = [5,6,7]
labels_of_interest_mapillary  = [44,45,46,47,48,49,50]


# inferencer = MMSegInferencer(model='deeplabv3plus_r50-d8_4xb2-300k_mapillay_v1_65-1280x1280',device='cuda:0')
# inferencer = MMSegInferencer(model='deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024',device='cuda:0')
# inferencer = MMSegInferencer(model='deeplabv3plus_r101b-d8_4xb2-80k_cityscapes-769x769',device='cuda:0')
# inferencer = MMSegInferencer(model='segformer_mit-b5_8xb1-160k_cityscapes-1024x1024',device='cuda:0')
# inferencer = MMSegInferencer(model='upernet_r101_4xb2-80k_cityscapes-769x769',device='cuda:0')


def processMmSegmentation(np_image, labels_of_interest, outputfilename):
    result = inferencer(np_image,return_datasamples=True)
    seg_mask = np.uint8(result.pred_sem_seg.data[0].detach().cpu())
    seg_mask[~np.any(seg_mask == np.array(labels_of_interest)[:, None, None], axis = 0)] = 0       # mask only value 5,6,7 voor cityscapes, mapillary 44,45,46,47,48,49,50
    seg_mask[seg_mask > 0] = 1
    #_, labels = cv2.connectedComponents(seg_mask)
    #cv2.imwrite(outputfilename + '_mask.png', labels)
    
    seg_img = Image.fromarray(seg_mask*255)
    np_image = Image.fromarray(np_image) #np_image[:, :, ::-1]
    np_image.paste(seg_img, (0, 0), seg_img)
    np_image.save(outputfilename)

# image = Image.open('horus_media_examples/GeoAI/input/img_01120_1_ab7f2e61-a1ba-4c61-81ae-81a136cc54bb.png')
# np_img = np.array(image)
# #cv2_img = cv2.imread('horus_media_examples/GeoAI/input/img_01120_1_ab7f2e61-a1ba-4c61-81ae-81a136cc54bb.png')
# processMmSegmentation(np_img, labels_of_interest_cityscapes, 'horus_media_examples/GeoAI/output/mmsegmentation.png')

# count = 0
# images = []
# for root, dirs, files in os.walk(os.path.abspath("C:/Horus/Horus Input en Data/data_scale1_shortclip/")):
#     for file in files:
#         images.append(os.path.join(root, file))
# for image in images:
#     img = cv2.imread(image)
#     #img = img[:, :, ::-1] #BGR to RGB
#     processMmSegmentation(img, labels_of_interest_cityscapes, 'horus_media_examples/GeoAI/output/' + str(count) + '.png')
#     count += 1
