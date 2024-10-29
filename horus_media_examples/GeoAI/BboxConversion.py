from tarfile import PAX_FIELDS
from PIL import Image, ImageDraw
from GeoAI.BBoxResult import BBoxResult

def convert_bboxes_Yolo(result):
    bboxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bboxes.append(BBoxResult(x1=x1,y1=y1,x2=x2,y2=y2,conf=float(box.conf[0]),classID=int(box.cls[0]) + 1))
    return bboxes

def convert_bboxes_Owl2(boxes, confs, labels):
    bboxes = []
    boxes =  boxes.cpu().numpy()
    confs =  confs.cpu().numpy()
    labels= labels.cpu().numpy()
    for box,conf,label in zip(boxes,confs,labels):
        x1, y1, x2, y2 = box[0],box[1],box[2],box[3]
        bboxes.append(BBoxResult(x1=x1,y1=y1,x2=x2,y2=y2,conf=float(conf),classID=int(label)))
    return bboxes

def to_bbox_centroids(bboxes):
    points = []
    for box in bboxes:
        Px = (box.x1 + box.x2) / 2
        Py = (box.y1 + box.y2) / 2
        points.append(zip(Px,Py))
    return points

def to_bbox_GroundPoints(bboxes):
    points = []
    for box in bboxes:
        Px = (box.x1 + box.x2) / 2
        Py = box.y2
        points.append((Px,Py))
    return points

def to_bbox_from_bboxResults(bboxresult_List):
    points = []
    for box in bboxresult_List:
        points.append([box.x1,box.y1,box.x2,box.y2])
    return points  

def draw_bboxes(pil_image, bboxes):
    draw = ImageDraw.Draw(pil_image)
    for bbox in bboxes:
        draw.rectangle(xy=((bbox.x1, bbox.y1), (bbox.x2, bbox.y2)), outline="red")
        draw.text(xy=(bbox.x1, bbox.y1), text=bbox.className + str(round(bbox.conf, 5)))
    pil_image.show()

def draw_Points(pil_image, points):
    draw = ImageDraw.Draw(pil_image)
    for pt in points:
        x,y = pt
        draw.rectangle(xy=((x-5, y-5), (x+5, y+5)), outline="red", fill="red")
    pil_image.show()