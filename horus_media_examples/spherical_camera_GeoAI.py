import psycopg2
import sys
import numpy as np
import logging
import io
from PIL import Image
import cv2
import torch
#torch.set_grad_enabled(False)

from horus_media import Client, ImageRequestBuilder, ImageProvider, Size, Mode, Grid, Scales
from horus_camera import (
    SphericalCamera,
    Pixel,
    SphericalImage,
    GeoReferencedPixel,
    ViewParameterizedPixel,
)
from horus_db import Frames, Recordings, Frame, Recording
from horus_gis import SchemaProvider
from horus_geopandas import HorusGeoDataFrame
from Connection_settings import connection_settings, connection_details, get_database_connection_string

# from GeoAI.OpticalFlow.ScaleRaft.main import processScalableRaft as ScalableRaft
# from GeoAI.OpticalFlow.ScaleRaft.main import loadScalableRaftModel
# from GeoAI.OpticalFlow.ScaleRaft.main import save_flow_to_arrow_image

# from GeoAI.Detection.VLM_Grounding.GroundingDinoLocal import processGroundingDino_All
# from GeoAI.Detection.VLM_Grounding.GroundingDinoPipeline import processGroundingDinoPipeline_All
# from GeoAI.Detection.VLM_Grounding.GroundingOwl2Transformers import processOwl2_All
from GeoAI.Detection.VLM_Grounding.GroundingYoloWorldUltralytics import processYoloWorld
# from GeoAI.Detection.VLM_Grounding.GroundedYoloWorldInference import processYoloWorld_All
# from GeoAI.Detection.VLM_Grounding.GroundingFlorence2 import processFlorence2Grounding_All
# from GeoAI.Segmentation.Pretrained.Transformers import processTransformers_All
# from GeoAI.Segmentation.Pretrained.mmSegmentation import processMmSegmentation
# from GeoAI.Segmentation.VLM_Grounding_Segment.GroundingDinoWithSAM import processDinoGroundedSam_All
# from GeoAI.Segmentation.VLM_Grounding_Segment.MerveGroundingDinoWithSAM import processMerveDinoGroundedSam_All
# from GeoAI.Segmentation.VLM_Grounding_Segment.GroundingLangSAM import processGroundedLangSam_All
# from GeoAI.Segmentation.VLM_Grounding_Segment.GroundingFlorence2WithFlorence2 import processFlorence2Segment_All
# from GeoAI.Segmentation.NonGrounded_Segment.SAM_V1_Local import processSAM_V1_All
# from GeoAI.Segmentation.NonGrounded_Segment.ExpeditSAM_Local import processExpeditSAM_All
# from GeoAI.Segmentation.NonGrounded_Segment.SlimSAM_transformers import processSlimSam_All
# from GeoAI.Segmentation.NonGrounded_Segment.SAM_HQ_Local import processSAM_HQ_All
# from GeoAI.Segmentation.NonGrounded_Segment.SAM_V2_ONNX import processSAM_V2_onnx_All
from GeoAI.Segmentation.NonGrounded_Segment.SAM_V2_Ultralytics import processSAM_V2_Ultralytics_All

from GeoAI.BboxConversion import convert_bboxes_Yolo, draw_bboxes, draw_Points, to_bbox_GroundPoints, to_bbox_from_bboxResults

import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point

pd.set_option("display.precision", 10)

connect_details = connection_settings(**connection_details)                     # Get connection info
print("user", connect_details)
connection = psycopg2.connect(get_database_connection_string())
try:
    client = Client("http://192.168.6.100:5050/web/", timeout=20)
    client.attempts = 5
    print("client retrieved")
except OSError as exception:
    logging.error(f"{exception}. Connecting to server 'http://192.168.6.100:5050/web/")
    sys.exit(1)
recordings = Recordings(connection)

front_camera = SphericalCamera()
back_camera  = SphericalCamera()
right_camera = SphericalCamera()
left_camera  = SphericalCamera()

def setup_Cameras():                                        # Step 1. create and configure spherical camera                                                 
    front_camera.set_network_client(client)
    front_camera.set_horizontal_fov(float(110))
    front_camera.set_yaw(float(0))
    front_camera.set_pitch(float(0))
    back_camera.set_network_client(client)
    back_camera.set_horizontal_fov(float(110))
    back_camera.set_yaw(float(180))
    back_camera.set_pitch(float(0))
    right_camera.set_network_client(client)
    right_camera.set_horizontal_fov(float(110))
    right_camera.set_yaw(float(90))
    right_camera.set_pitch(float(0))
    left_camera.set_network_client(client)
    left_camera.set_horizontal_fov(float(110))
    left_camera.set_yaw(float(270))
    left_camera.set_pitch(float(0))

def list_recordings():
    cursor = recordings.all()
    recording = Recording(cursor)
    while recording is not None:
        print(" ", recording.id, " ", recording.directory)
        recording = Recording(cursor)

def get_and_save_recordingPoint_to_json(recordingID):
    frames = Frames(connection)
    cursor = frames.query(recordingid=recordingID)
    frame = Frame(cursor)
    dict = {'Index':[],'recordingID':[],'uuID':[],'lon':[],'lat':[],'heading':[]}
    df = pd.DataFrame(dict)
    while frame is not None:
        #print(" ",frame.recordingid," ",frame.uuid," lon: ",frame.longitude," lat: ",frame.latitude," heading: ",frame.azimuth, " index: " + str(frame.index))
        localDf = {'Index':frame.index,'recordingID':frame.recordingid,'uuID':frame.uuid,'lon':frame.longitude,'lat':frame.latitude,'heading':frame.azimuth}
        df = df._append(localDf, ignore_index = True)
        frame = Frame(cursor)
    #display(df)
    df['geometry'] = df.apply(lambda x: Point((float(x.lon), float(x.lat))), axis=1)
    df = GeoDataFrame(df, geometry='geometry')
    df.to_file('recording' + str(recordingID) + '.geojson', driver='GeoJSON') 

def get_and_save_image_file(recording, frame, camera, name):
    camera.set_frame(recording, frame)
    image = camera.acquire(Size(1920,1080))
    with open("./output/" + str(frame.index) + name + ".png", "wb") as image_file:
            image_file.write(image.get_image().getvalue())
            image.get_image().close()

def get_and_save_North_image_file(recording, frame, camera, name):
    camera.set_yaw(-frame.azimuth)
    #camera.set_pitch(float(0))
    camera.set_frame(recording, frame)
    image = camera.acquire(Size(1920,1080))
    with open("./output/" + str(frame.index) + name + ".png", "wb") as image_file:
        image_file.write(image.get_image().getvalue())
        image.get_image().close()

sp = SchemaProvider()
database = HorusGeoDataFrame(sp.single_measurement())

geoLocationDict = {'Index':[],'recordingID':[],'uuID':[],'lon':[],'lat':[],'heading':[]}
locations_car   = pd.DataFrame(geoLocationDict)
geoPointDict = {'Index':[],'recordingID':[],'uuID':[],'lon':[],'lat':[],'alt':[]}
locations_pole   = pd.DataFrame(geoPointDict)


def add_imgPixels_as_geoPixels(sphere_image,imgPixel, index, recordingid, uuid):
    for pt in imgPixel:
        x,y = pt
        if (y <= 540):
            print ('Skipping ' +str(x) + ',' + str(y) + ' because of pitch constraint')
            continue
        geoPixel: GeoReferencedPixel = sphere_image.project_pixel_on_ground_surface(Pixel(col=x,row=y))
        locations_pole.loc[len(locations_pole.index)] = [index,recordingid,uuid,geoPixel.geo_location.lon,geoPixel.geo_location.lat,geoPixel.geo_location.alt]


def get_spherical_camera_frames(recordingID):
    recording = next(Recording.query(recordings, recordingid=recordingID))
    recordings.get_setup(recording)
    frames = Frames(connection)                                                     # Step 2. Get a recorded frame and place the camera onto that 'frame'
    results = Frame.query(frames,recordingid=recordingID,order_by="index")

    with torch.no_grad():
        #model = ScalableRaftModel()
        for index, t in enumerate(results):
            results = Frame.query(frames, recordingid=t.recordingid, index=t.index, order_by="index",)
            frame = next(results)
            if frame is None: 
                print("No frames!") 
                sys.exit()
            
            front_camera.set_frame(recording, frame)
            #front_camera.set_pitch(float(-20.0))
            sphere_image = front_camera.acquire(Size(1920,1080))
            
            bytes_img = sphere_image.get_image().getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_img, np.uint8), 1)
            pil_image = Image.open(io.BytesIO(bytes_img))
            np_img = np.array(pil_image)
            npcv2_img = np.array(cv2_img)
            outputFileName = "horus_media_examples/output/" + str(frame.index) + ".png"
            #processTransformers_All(pil_image,[5,6,7],outputFileName)
            #processMmSegmentation(np_img,[5,6,7],outputFileName)
            #processGroundingDino_All(pil_image,np_img,outputFileName,text="pole",outputFileName)
            #processGroundingDinoPipeline_All(pil_image=pil_image,text=["pole."],outputFileName=outputFileName)
            #processDinoGroundedSam_All(pil_image=pil_image,text=["pole."],np_image=np_img,outputFileName=outputFileName)
            #processMerveDinoGroundedSam_All(pil_image=pil_image,text=["pole."],np_image=np_img,outputFileName=outputFileName)
            #processMerveDinoGroundedSam_All(pil_image=pil_image,np_image=np_img,text= ["pole"],outputFileName=outputFileName)
            #processGroundedLangSam_All(pil_image=pil_image,text="pole",box_threshold=0.24,text_threshold=0.2,outputFileName=outputFileName)
            
            #processYoloWorld_All(pil_image=cv2_img,outputFileName=outputFileName)   # Search text is hard coded in class
            bboxResults = convert_bboxes_Yolo(processYoloWorld(pil_image=cv2_img))
            bboxes = to_bbox_from_bboxResults(bboxresult_List=bboxResults)
            #processSAM_V1_All(cv2_image=cv2_img,bboxes=bboxes,outputFilename=outputFileName)
            #processSlimSam_All(pil_image=pil_image,bboxes=bboxes,outputFileName=outputFileName)
            #processExpeditSAM_All(cv2_image=cv2_img,bboxes=bboxes,outputFileName=outputFileName)
            #processSAM_HQ_All(cv2_image=cv2_img,bboxes=bboxes,outputFileName=outputFileName)
            #processSAM_V2_onnx_All(np_image=npcv2_img,bboxes=bboxes,outputFileName=outputFileName)
            processSAM_V2_Ultralytics_All(cv2_image=cv2_img,bboxes=bboxes,outputFileName=outputFileName)

            #imgPixel = to_bbox_GroundPoints(bboxes=bboxResults)
            #add_imgPixels_as_geoPixels(sphere_image=sphere_image,imgPixel=imgPixel,index=frame.index,recordingid=frame.recordingid, uuid=frame.uuid)
            #draw_Points(pil_image=pil_image,points=imgPixel)
            #processFlorence2Grounding_All(pil_image=pil_image,outputFilename=outputFileName,text='pole')
            #processFlorence2Segment_All(pil_image=pil_image,outputFilename=outputFileName,text='pole')
            #processOwl2_All(pil_image=pil_image,outputFilename=outputFileName,text=['pole'])
            #processYoloWorld_All(pil_image=pil_image,outputFileName=outputFileName)

            # if (index > 0):
            #     flowdata = ScalableRaft(model,np_img_old,np_img)
            #     img_new = save_flow_to_arrow_image(np_img,flowdata)
            #     cv2.imwrite("./output/" + str(frame.index) + "_flow.png", img_new)
            # np_img_old = np_img
            
            print(frame.recordingid," index: " + str(frame.index)," ",frame.uuid," lon: ",frame.longitude," lat: ",frame.latitude," heading: ",frame.azimuth)
            # loc_car = {'Index':frame.index,'recordingID':frame.recordingid,'uuID':frame.uuid,'lon':frame.longitude,'lat':frame.latitude,'heading':frame.azimuth}
            # locaties_car = locaties_car._append(loc_car, ignore_index = True)

            #if (index > 500): break

            # pil_image.save("horus_media_examples/output/" + str(frame.index) + "_front_pil.png")
            # get_and_save_image_file(recording, frame,front_camera, "_front")
            # get_and_save_image_file(recording, frame,back_camera , "_back" )
            # get_and_save_image_file(recording, frame,right_camera, "_right")
            # get_and_save_image_file(recording, frame,left_camera , "_left" )
    # locaties_car['geometry'] = locaties_car.apply(lambda x: Point((float(x.lon), float(x.lat))), axis=1)
    # locaties_car = GeoDataFrame(locaties_car, geometry='geometry') #,crs="EPSG:28992"
    # locaties_car.to_file('Locations recording=' + str(recordingID) + '.geojson', driver='GeoJSON') 
          
    locations_pole['geometry'] = locations_pole.apply(lambda x: Point((float(x.lon), float(x.lat))), axis=1)
    gdflocations_pole = GeoDataFrame(locations_pole, geometry='geometry') #,crs="EPSG:28992"
    gdflocations_pole.to_file('Pole recording=' + str(recordingID) + '.geojson', driver='GeoJSON')   

if __name__ == "__main__":
    setup_Cameras()
    #list_recordings()
    #get_and_save_recordingPoint_to_json(5)
    get_spherical_camera_frames(5)     #4=goProMax, 5=Ladybug, 88=Rihad