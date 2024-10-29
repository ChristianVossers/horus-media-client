import numpy as np
import cv2
import torch
import pickle
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()        # use bfloat16 for the entire notebook
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
from sam2.build_sam import build_sam2_camera_predictor
import matplotlib.pyplot as plt
from MaskInfo import MaskInfo

# Parameters
MIN_AREA = 100                                  # Stop tracking if objects have less pixels than this.
MAX_AREA = 50000                                # Stop tracking if objects are very large because of detection or segmentation errors. Shouldnt be necessary if these work correct.
IOU_THRESHOLD = 0.5                             # Stop tracking if objects symmetric IoU is low. 0.5 means half-overlap of actual frame and tracked projection of previous frame.
#MAX_COMPONENTS = 4                              # Stop tracking if objects have many parts (fragmented)
MAX_TRAJECT_LENGTH = 9                          # Stop tracking if traject gets too long and is lingering. Warning: too short will lead to multiple detections of the same object.
SHOW_TRAIL_LENGTH = 4                           # Show maximum amount of trajectory trail although traject might be much longer. Else it all looks very cluttered.

# Tracker init
model_cfg = "tracker_large.yaml"
predictor = build_sam2_camera_predictor(model_cfg, "horus_media_examples/GeoAI/models/tracker_large.geoAI")
print('Memory Module loaded.')

# Persistent and local memory structures init
MaskInfos_archive = {}                                                  # All archived trajectories. Trajectories which are done and in the past.
MaskInfos_current = {}                                                  # All currently tracked trajectories. Keep tracking these.
objectCounter = 0                                                       # Nr of objects tracked overall. So archive plus current.
width = 0                                                               # Memory mask working size width
height = 0                                                              # Memory mask working size height
obj_ids_1 = mask_logits_1 = mask_list_1 = []                            # Initialize in case there are no existing or new objects.
obj_ids_2 = mask_logits_2 = mask_list_2 = []                            # Initialize in case there are no future objects.
obj_ids_1_tracked = mask_logits_1_tracked = mask_list_1_tracked = []    # Initialize in case there are no existing objects.



################################# Small universal functions ##################################
def calculate_area(mask):
    return np.count_nonzero(mask > 0)
def calculate_iou(mask1, mask2):
    mask1_area = np.count_nonzero( mask1 )
    mask2_area = np.count_nonzero( mask2 )
    intersection = np.count_nonzero( np.logical_and( mask1, mask2 ) )
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou
def generate_objCounter_id_names(bboxes):
    global objectCounter
    names = []
    for bbox in bboxes:
        names.append(str(objectCounter))
        objectCounter += 1
    return names
def generate_bbox_id_names(frameNr, bboxes):
    global objectCounter
    names = []
    for bbox in bboxes:
        names.append(str(frameNr) + '_' + str(bbox))
        objectCounter += 1
    return names


def too_small(area):
    return area < MIN_AREA
def too_big(area):
    return area > MAX_AREA
def too_long(times):
    return len(times) > MAX_TRAJECT_LENGTH

# def too_fragmented(mask):                                           # TODO Return the amount of connected components. Too many means fragmented.
#     nr_components, labels = cv2.connectedComponents(mask)
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image=mask,connectivity=8)
#     # TODO check stats to see if largest is still large enough.
#     # TODO Only keep parts that are large enough. Remove small particles off the mask.
#     #show_connected_components(labels)
#     if (nr_components > MAX_COMPONENTS): return True
#     return False
def getGroundPoint(mask):                                         # Find lowest representative mask point. Is used as a groundpoint for pole-like objects.
    rows = np.any(mask, axis=1)
    if not max(rows):
        return None
    _, y = np.where(rows)[0][[0, -1]]
    line_lowest_y = np.any(mask[y], axis=1)
    group_lowest_y = np.where(line_lowest_y != 0)
    x = int(np.average(group_lowest_y))

    # cv2_image_masked = cv2.cvtColor(mask*255, cv2.COLOR_GRAY2BGR)
    # cv2.circle(cv2_image_masked, (x,y), radius=3, color=(0,0,255), thickness=-1)
    # cv2.namedWindow('connected')
    # cv2.imshow('connected', cv2_image_masked)
    # cv2.moveWindow('connected', 0,0)
    # cv2.waitKey(200)
    return (x,y)

#################### DRAW AND SAVE HELPER FUNCTIONS (mostly for debug) ########################
def drawTrackedFrameNP(name, cv2_image, masks):
    final_mask = np.zeros((height, width, 1), dtype=np.uint8)                             # start with a empty 2D [0,0,0...0] mask
    for mask in masks:
        final_mask = cv2.bitwise_or(final_mask, mask)
    #cv2_image_masked = cv2.bitwise_and(cv2_image,cv2_image,mask=final_mask)
    cv2_image_masked = cv2.addWeighted(cv2_image, 0.5, cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR), 164, 0)
    #cv2.imwrite('output.png',cv2_image_masked)
    cv2.namedWindow(name)
    cv2.imshow(name, cv2_image_masked)
    cv2.moveWindow(name, 0,0)
    cv2.waitKey(1000)  
def drawTrackedFrame(cv2_image, mask_logits):
    mask = np.zeros((height, width, 1), dtype=np.uint8)                             # start with a empty 2D [0,0,0...0] mask
    for mask_logit in mask_logits:
        out_mask = (mask_logit > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        mask = cv2.bitwise_or(mask, out_mask)

    cv2_image_masked = cv2.bitwise_and(cv2_image,cv2_image,mask=mask)
    cv2.imshow("Masks", cv2_image_masked)
    cv2.waitKey()  
def drawTrackedFrames(cv2_image, mask_logitsList, boxesList, obj_idList):
    final_mask = np.zeros((height, width), dtype=np.uint8)

    for _, orig_mask_logits in enumerate(mask_logitsList):  
        mask = np.zeros((height, width, 1), dtype=np.uint8)                             # start with a empty 2D [0,0,0...0] mask
        for old_mask_logit in orig_mask_logits:
            out_mask = (old_mask_logit > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 84

            mask = cv2.bitwise_or(mask, out_mask)
            # cv2.imshow("Masks", final_mask)
            # cv2.waitKey()
        final_mask = np.maximum(final_mask,mask)
    

    draw_cv2_image = cv2.addWeighted(cv2_image, 1, cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR), 2, 0)
    
    for j in range(len(boxesList)):
        boxes   = boxesList[j]
        obj_ids = obj_idList[j]
        for i in range(len(boxes)):
            bbox = boxes[i]
            obj_id = obj_ids[i]
            cv2.rectangle(draw_cv2_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0,0,255), thickness=1)
            cv2.putText(draw_cv2_image,obj_id, (bbox[0],bbox[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4,(0,0,255),1)
    cv2.imshow("Masks", draw_cv2_image)
    #cv2.imwrite("" + str(count)+".png",draw_cv2_image)
    cv2.waitKey()
def drawTensors(tensor_logits):
    for tensor_logit in tensor_logits:
        np_logit = tensor_logit.permute(1, 2, 0).cpu().numpy().astype(np.uint8)     # (tensor_logit > 0.0) betekend minimum waarde per pixel. Hoger = 'minder vette tekening' van een lichtmast
        plt.imshow(np_logit)
        plt.show()
def drawObjectInfo(cv2_image, objectinfo):
    masks_current   = np.zeros((height, width, 1), dtype=np.uint8)
    masks_projected = np.zeros((height, width, 1), dtype=np.uint8)
    masks_masklist  = np.zeros((height, width, 1), dtype=np.uint8)
    masks_current   = cv2.bitwise_or(masks_current, objectinfo.mask)
    len_mask_list_last = len(objectinfo.mask_list)
    len_mask_list_first= max(0,len_mask_list_last-5)
    for index in range(len_mask_list_first,len_mask_list_last):
        masks_masklist = cv2.bitwise_or(masks_masklist, objectinfo.mask_list[index])
    if objectinfo.mask_tracked is not None:
        masks_projected = cv2.bitwise_or(masks_projected, objectinfo.mask_tracked)
    mask_final = cv2.merge((masks_projected,masks_masklist,masks_current))
    cv2_image_masked = cv2.addWeighted(cv2_image, 0.7, mask_final*255, 3, 0)
    cv2.imshow("Objectinfos", cv2_image_masked)
    cv2.moveWindow("Objectinfos", 2200, 0) 
    cv2.waitKey(20)
def drawObjectInfos(cv2_image, objectinfos):
    masks_current   = np.zeros((height, width, 1), dtype=np.uint8)
    masks_projected = np.zeros((height, width, 1), dtype=np.uint8)
    masks_masklist  = np.zeros((height, width, 1), dtype=np.uint8)
    for objectinfo in objectinfos.values():
        masks_current   = cv2.bitwise_or(masks_current, objectinfo.mask)
        #cv2.rectangle(masks_current,(objectinfo.bbox[0],objectinfo.bbox[1]),(objectinfo.bbox[2],objectinfo.bbox[3]),255,2)         # Add Bbox rect
        # if objectinfo.bbox is not None:                                                                                           # Add Area text
        #     cv2.putText(masks_current, str(objectinfo.area), (objectinfo.bbox[0] - 20,objectinfo.bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 1, cv2.LINE_AA)
        # len_mask_list_last = len(objectinfo.history_mask)
        # len_mask_list_first= max(0,len_mask_list_last-SHOW_TRAIL_LENGTH)
        # for index in range(len_mask_list_first,len_mask_list_last):
        #     #masks_masklist = cv2.bitwise_or(masks_masklist, objectinfo.history_mask[index])
        #     contours, _ = cv2.findContours(objectinfo.history_mask[index], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #     cv2.drawContours(masks_masklist, contours, -1, (1),1)

        for index in range(1,len(objectinfo.history_groundpoint)):
            cv2.circle(masks_masklist, objectinfo.history_groundpoint[index], radius=3, color=1, thickness=-1)
            cv2.line(masks_masklist,objectinfo.history_groundpoint[index-1],objectinfo.history_groundpoint[index],1,1)
        if objectinfo.mask_tracked is not None:
            masks_projected = cv2.bitwise_or(masks_projected, objectinfo.mask_tracked)
    mask_final = cv2.merge((masks_projected,masks_masklist,masks_current))      # B=Tracked mask, G=History masks, R=Current mask
    return cv2.addWeighted(cv2_image, 0.7, mask_final*255, 3, 0)
def showObjectInfos(cv2_image, objectinfos):
    cv2_image_masked = drawObjectInfos(cv2_image, objectinfos)
    cv2.imshow("Objectinfos", cv2_image_masked)
    cv2.moveWindow("Objectinfos", 0, 50) 
    cv2.waitKey(20)
def saveObjectInfos(filename, cv2_image, objectinfos):
    cv2_image_masked = drawObjectInfos(cv2_image, objectinfos)
    cv2.imwrite(filename=filename,img=cv2_image_masked)   
def show_connected_components(label_mask):
    label_hue = np.uint8(179*label_mask/np.max(label_mask))             # Map component labels to hue val
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)          # cvt to BGR for display
    labeled_img[label_hue==0] = 0                                       # set bg label to black
    cv2.imshow('connected',labeled_img)
    cv2.waitKey()

################################# Transition helper functions #################################
def MasktoBbox(single_mask):
    rows = np.any(single_mask, axis=1)
    cols = np.any(single_mask, axis=0)
    if (not rows.max()) or (not cols.max()):                    # No BB found (for instance tracked out of image)
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    #if (cmin, rmin, cmax, rmax) == (0,0,0,0): return None       # Return None if no track found
    return cmin, rmin, cmax, rmax
def TensorstoBbox(tensor_logits):
    logit_boxes = []
    for tensor_logit in tensor_logits:
        np_logit = (tensor_logit > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)     # >x betekend minimum waarde per pixel. Hoger = 'minder vette tekening' van een lichtmast
        logit_boxes.append(MasktoBbox(np_logit))
    return logit_boxes
def LogitsToMasks(mask_logits):
    masks = []                             # start with a empty 2D [0,0,0...0] mask
    for mask_logit in mask_logits:
        masks.append((mask_logit > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    return masks


######################################## ADD TO MEMORY ########################################
def add_mask_to_memory(mask, obj_id):
     return predictor.add_new_mask(frame_idx=0, obj_id=obj_id, mask=mask) 
def add_boxes_to_memory(bboxes, obj_ids):
    for index, bbox in enumerate(bboxes):
        _, new_obj_ids, new_mask_logits = predictor.add_new_prompt(frame_idx=0, obj_id=obj_ids[index], bbox=np.array([bbox], dtype=np.float32))
    return new_obj_ids, new_mask_logits
def add_objectsinfos_to_memory(objectinfos):
    for objectinfo in objectinfos.values():
        if objectinfo.mask_tracked is None: continue
        _, new_obj_ids, new_mask_logits = add_mask_to_memory(mask=np.squeeze(objectinfo.mask_tracked), obj_id=objectinfo.id)
    return new_obj_ids, new_mask_logits  


######################################## Memory runner ########################################
def init(workingSize):
    global width, height
    global obj_ids_1,mask_logits_1,mask_list_1
    global obj_ids_2,mask_logits_2,mask_list_2
    global obj_ids_1_tracked,mask_logits_1_tracked,mask_list_1_tracked 
    width = workingSize[1]
    height = workingSize[0]
    obj_ids_1 = mask_logits_1 = mask_list_1 = []                            # Initialize in case there are no existing or new objects.
    obj_ids_2 = mask_logits_2 = mask_list_2 = []                            # Initialize in case there are no future objects.
    obj_ids_1_tracked = mask_logits_1_tracked = mask_list_1_tracked = []
    

def runFrame(frameNr, cv2_image_1, cv2_image_2, boxes_1, boxes_2):

    global obj_ids_1,mask_logits_1,mask_list_1
    global obj_ids_2,mask_logits_2,mask_list_2
    global obj_ids_1_tracked,mask_logits_1_tracked,mask_list_1_tracked 
    

    ################# SAM OLD & ADD PREVIOUS ###############
    predictor.load_first_frame(cv2_image_1)

    if len(boxes_1) > 0:
        obj_ids_1, mask_logits_1 = add_boxes_to_memory(bboxes=boxes_1,obj_ids=generate_objCounter_id_names(bboxes=boxes_1))
    if len(MaskInfos_current) > 0: 
        obj_ids_1, mask_logits_1 = add_objectsinfos_to_memory(objectinfos=MaskInfos_current)
    if len(obj_ids_1) > 0:
        mask_list_1 = LogitsToMasks(mask_logits_1)

    if len(mask_list_1)==0 or predictor._get_obj_num() != len(mask_list_1):  # Edge case: only object is deleted because bad object (big/small...), still in tracker memory.
        print('no current tracks or all deleted in previous round')
        return

    ######################## TRACK ########################
    obj_ids_1_tracked, mask_logits_1_tracked = predictor.track(cv2_image_2)
    mask_list_1_tracked = LogitsToMasks(mask_logits_1_tracked)

    ###################### SAM NEW ########################
    predictor.load_first_frame(cv2_image_2)
    if len(boxes_2) > 0:
        obj_ids_2, mask_logits_2 = add_boxes_to_memory(boxes_2, generate_objCounter_id_names(bboxes=boxes_2))
        mask_list_2 = LogitsToMasks(mask_logits_2)
    MaskInfo_dict2 = {}
    for index2, obj_id2 in enumerate(obj_ids_2):
        maskinfo2 = MaskInfo(id=obj_id2, mask=mask_list_2[index2], id_hist=obj_id2, timestamp=frameNr - 1)
        maskinfo2.bbox = MasktoBbox(mask_list_2[index2])
        MaskInfo_dict2[obj_id2] = maskinfo2

    for index, obj_id in enumerate(obj_ids_1):
        ###################### MEMORY: OLD ########################

        maskInfo_1_area= calculate_area(mask_list_1[index])                     # Calculate pixel area
        if obj_id in MaskInfos_current:                                         # Object was alrerady tracked and added with add_objectsinfos_to_memory
            maskInfo_1 = MaskInfos_current[obj_id]
            maskInfo_1.id_hist = maskInfo_1.id_hist + ' (' + str(maskInfo_1_area) + ') =>.'
        else:                                                                   # Object was new loaded from bboxes1 and added with add_boxes_to_memory
            maskInfo_1 = MaskInfo(id=obj_id,id_hist=obj_id)
            maskInfo_1.id_hist = maskInfo_1.id_hist + ' (' + str(maskInfo_1_area) + ')'

        maskInfo_1.mask=mask_list_1[index]
        maskInfo_1.mask_tracked=mask_list_1_tracked[index]
        maskInfo_1.history_mask.append(maskInfo_1.mask)
        maskInfo_1.timestamp=frameNr
        maskInfo_1.bbox = MasktoBbox(maskInfo_1.mask)
        maskInfo_1.area = maskInfo_1_area                                      # Give pixel area
        maskInfo_1.history_area.append(maskInfo_1.area)
        maskInfo_1.history_timestamp.append(maskInfo_1.timestamp)
        maskInfo_1.bbox_tracked = MasktoBbox(maskInfo_1.mask_tracked)
        maskInfo_1tracked_area = calculate_area(maskInfo_1.mask_tracked)
        maskInfo_1groundPoint = getGroundPoint(maskInfo_1.mask)
        
        ###################### CONTINUE TRAJECT ########################
        # Check if tracked area is too small. Then don't add to mask_tracked but add as a final of mask_list (and archive)    
        if  not too_small(maskInfo_1tracked_area) and \
            not too_big(maskInfo_1tracked_area) and \
            not too_long(maskInfo_1.history_timestamp) and \
            maskInfo_1groundPoint is not None:
                
            maskInfo_1.history_groundpoint.append(maskInfo_1groundPoint)

            # TODO Check if NOT TOO CLOSE to left and right edge of the screen. Stop tracking if overlaps 'significantly'
            # TODO Check if area jumps up to high or low. Notsure: might not add to mask_tracked but add as a final of mask_list (and archive)
            # TODO Find outliers in GroundPoint data.
            # TODO Find outliers in WGS84 projected positioning data. Huge leaps? Probably tracking or detection failed. Mark positition to not be included in triangulation or positioning.
            # TODO make history_sam2confidence and history_detectorconfidence

            ################### MEMORY: COMPARE TO NEW #####################
            isMatched = False
            for id2 in MaskInfo_dict2.keys():
                maskInfo_2 = MaskInfo_dict2[id2]
                iou = calculate_iou(maskInfo_1.mask_tracked, maskInfo_2.mask)
                if iou > IOU_THRESHOLD:
                    maskInfo_1.id_hist = maskInfo_1.id_hist + '=>' + maskInfo_2.id
                    maskInfo_1.mask_tracked = maskInfo_2.mask
                    maskInfo_1.bbox_tracked = MasktoBbox(maskInfo_1.mask_tracked)
                    maskInfo_1.area = calculate_area(maskInfo_2.mask)    
                    isMatched = True
            
            ################## MEMORY: STORE OR CONTINUE ###################
            MaskInfos_current[obj_id] = maskInfo_1                              # Track on. Add for upcoming frame

            ################## PRETTY PRINT TRAJECTORY CMD #################
            if (isMatched): print (str(index) + ' match:     ' + maskInfo_1.id_hist)   
            else:
                if len(maskInfo_1.history_area) == 1:
                            print (str(index) + ' start:     ' + maskInfo_1.id_hist)
                else:       print (str(index) + ' continue:  ' + maskInfo_1.id_hist)
        else:                                                                   # Track is done:
            ################## STOP AND ARCHIVE TRAJECT ####################
            MaskInfos_archive[obj_id] = maskInfo_1                              # Move to archive
            MaskInfos_current.pop(obj_id,None)                                  # Remove from current tracking
            print (str(index) + ' end:       ' + maskInfo_1.id_hist)
        showObjectInfos(cv2_image=cv2_image_1,objectinfos=MaskInfos_current)
