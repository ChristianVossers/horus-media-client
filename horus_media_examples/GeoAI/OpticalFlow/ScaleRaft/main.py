# import sys
# sys.path.append('core')
import cv2
from glob import glob
import argparse
import numpy as np
import torch
from PIL import Image

from .core.utils.flow_viz import flow2rgb
from .core.utils import frame_utils
from .core.raft_cscv import  RAFT343used
from .core.utils.utils import InputPadder
# from core.utils.flow_viz import flow2rgb
# from core.utils import frame_utils
# from core.raft_cscv import RAFT343used
# from core.utils.utils import InputPadder


def getDefaultArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default='./horus_media_examples/GeoAI/models/ScalableRAFT.geoAI')
    # parser.add_argument('--model', help="restore checkpoint", default='../../models/ScalableRAFT.geoAI')
    
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--num_heads', default=1, type=int, help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true', help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true', help='use position and content-wise attention')
    parser.add_argument('--start', default=0, type=int, help='where to start')
    return parser.parse_args()

def loadScalableRaftModel():
    with torch.inference_mode():
        model = torch.nn.DataParallel(RAFT343used(getDefaultArgs()))
        pretrained_dict = torch.load(str(getDefaultArgs().model))
        old_list = {}
        for k, v in pretrained_dict.items():
            old_list.update({k: v})
        model.load_state_dict(old_list, strict=False)
        model.cuda()
        model.eval()
        print ("ScaleRaft model loaded...")
        return model

def save_flow_to_arrow_image(img_orig, flow_uv):
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    height = flow_uv.shape[0]
    width = flow_uv.shape[1]
    for x in range(0,width,5):
        for y in range(0,height,5):
            dx = u[y][x]
            dy = v[y][x]
            if (abs(dx) + abs(dy)) > 10:
                cv2.line(img_orig,(x,y),(int(x+dx),int(y+dy)),(0, 0, 255),1)
                cv2.circle(img_orig,(int(x+dx),int(y+dy)),1,(255,0,0))
    return img_orig

def processScalableRaft(model, img1,img2, iters=12):
    with torch.inference_mode():
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        image1 = img1[None].cuda()
        image2 = img2[None].cuda()
        padder = InputPadder(image1.shape, mode='kitti',sp=16)
        image1, image2 = padder.pad(image1, image2)

        #res = gma_forward(image1, image2)
        #flow_pr = res['flow_preds'][0]
        flow_low, flow_pr, dchange = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).detach().cpu()
        flowdata = flow.permute(1, 2, 0).numpy()
        return flowdata

def process_standalone(path1,path2, iters=12):
    img1 = frame_utils.read_gen(path1)
    img2 = frame_utils.read_gen(path2)

    pathsplit = path1.split('/')
    idout = pathsplit[-1].split('.')[0]

    img1 = np.array(img1).astype(np.uint8)[..., :3]
    img2 = np.array(img2).astype(np.uint8)[..., :3]
    img1 = cv2.resize(img1, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    img2 = cv2.resize(img2, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
    image1 = img1[None].cuda()
    image2 = img2[None].cuda()
    padder = InputPadder(image1.shape, mode='kitti',sp=16)
    image1, image2 = padder.pad(image1, image2)

    #res = gma_forward(image1, image2)
    #flow_pr = res['flow_preds'][0]
    flow_low, flow_pr, dchange = model(image1, image2, iters=iters, test_mode=True)
    flow = padder.unpad(flow_pr[0]).detach().cpu()

    # dchange = padder.unpad(dchange[0,0]).detach().cpu().numpy()
    # frame_id = idout + 'depth_change.png'
    # datamin = np.min(dchange)
    # datamax = np.max(dchange)
    # mid_data = (datamin + datamax) * 0.5
    # lenthmid = 1 / (mid_data - datamin)
    # dchange = ((dchange - mid_data) * lenthmid).clip(-1, 1) * 128 + 128
    # dchange = ((dchange - 1)*16).clip(-1, 1) * 128 + 128
    # colormap = plt.get_cmap('plasma')  # plasma viridis
    # heatmap = (colormap((dchange).astype(np.uint8)) * 2 ** 8).astype(np.uint16)[:, :, :3]
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    # cv2.imwrite('%s/%s' % (outpath, frame_id), heatmap * 255)

    #frame_idf = idout + 'flow.png'
    flowdata = flow.permute(1, 2, 0).numpy()
    flowviz = (flow2rgb(flowdata) * 255).astype(np.uint8)
    flowviz = cv2.cvtColor(flowviz, cv2.COLOR_RGB2BGR)
    #cv2.imwrite('%s/%s' % (outpath, frame_idf), flowviz)

    img_new = save_flow_to_arrow_image(flowviz,flowdata)
    cv2.imwrite("scalableRaft.png", img_new)


if __name__ == '__main__':
    with torch.no_grad():
        model = loadScalableRaftModel()
        fileList = glob("C:/Horus/Horus Media Client/horus_media_examples/GeoAI/input/*.png")
        for cnt in range(1,len(fileList)):

            img1 = np.array(Image.open(fileList[cnt-1]))
            img2 = np.array(Image.open(fileList[cnt]))
            flowdata = processScalableRaft(model, img1,img2, iters=12)

            flowviz = (flow2rgb(flowdata) * 255).astype(np.uint8)
            flowviz = cv2.cvtColor(flowviz, cv2.COLOR_RGB2BGR)
            cv2.imwrite('scalableRaft_flo.png', flowviz)
            img_new = save_flow_to_arrow_image(flowviz,flowdata)
            cv2.imwrite("scalableRaft.png", img_new)
