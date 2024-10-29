import os
import numpy as np
import cv2
import MemoryModule as memory

memory.init(workingSize=(1080,1920))

if __name__ == '__main__':
    dir_images = "C:/Horus/Horus Input en Data/Fov110_front_Rdam Volgnummers_2/"
    file_images = []
    for root, dirs, files in os.walk(os.path.abspath(dir_images)):
        for file in files:
            file_images.append(file)
    dir_bboxes = "C:/Horus/Horus Input en Data/Fov110_front_Rdam Volgnummers_2_npy/"
    file_bboxes = []
    for root, dirs, files in os.walk(os.path.abspath(dir_bboxes)):
        for file in files:
            file_bboxes.append(file)

    memory.init(workingSize=cv2.imread(dir_images + file_images[0]).shape)

    #for count in range(len(file_images)-1,1,-1):
    for frameNr in range(800,1,-1):

        ###################### LOAD IMAGES #####################
        cv2_image_1 = cv2.imread(dir_images + file_images[frameNr])
        cv2_image_2 = cv2.imread(dir_images + file_images[frameNr - 1])
        boxes_1 = np.load(dir_bboxes + file_bboxes[frameNr])
        boxes_2 = np.load(dir_bboxes + file_bboxes[frameNr - 1])

        memory.runFrame(frameNr=frameNr, cv2_image_1=cv2_image_1,cv2_image_2=cv2_image_2,boxes_1=boxes_1,boxes_2=boxes_2)

        print('--------------------------------------------------------- objects in current memory: ' + str(len(memory.MaskInfos_current)) + ', objects in archive memory: ' + str(len(memory.MaskInfos_archive)))


        #################### DRAW OR SAVE ####################
        #memory.saveObjectInfos(filename='./output/frame ' + str(frameNr) + '.png',cv2_image=cv2_image_1,objectinfos=memory.MaskInfos_current)
