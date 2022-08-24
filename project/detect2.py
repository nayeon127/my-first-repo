import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
# from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from collections import Counter
from mediapipe_model import *
from utils2 import *
from object_select import *

# 개발용
def detect():
    source, weights, view_img, save_txt, imgsz, trace, save_path = args.source, args.weights, args.view_img, args.save_txt, args.img_size, not args.no_trace, args.save_path
    save_img = not args.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
  
    # Initialize
    set_logging()
    device = select_device(args.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)# load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, args.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names
    names = ['book','pen(pencil)','laptop','tabletcomputer','keyboard','cellphone','mouse','pencilcase','wallet','desklamp','airpods','stopwatch']
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    # Gaze estimation model with mediapipe
    gaze_model = GazeEstimation(args)
    object_select_model = ObjectSelect(args)

    t0 = time.time()
    first_cnt=0
    
    # save_path ='run.mp4'
    for path, img, im0s, vid_cap in dataset:
        
        first_cnt+=1
        if webcam:
            im0s = im0s[0]
            im0f = im0s.copy()
        else:
            im0f = im0s
        if first_cnt==1:
            if vid_cap:
                w = round(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = round(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
            else:
                fps, w, h = 1, im0s.shape[1], im0s.shape[0]    
            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        
        # with mp_face_mesh.FaceMesh( max_num_faces=1, refine_landmarks=True,
        # min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            
        # Draw the image.
        im0f.flags.writeable = False
        im0f = cv2.cvtColor(im0f, cv2.COLOR_BGR2RGB)
        # Load the face mesh model.
        results = gaze_model.faceMeshLoad(im0s)
        x = im0f.shape[1] # height
        y = im0f.shape[0] # width

        # book_head = y*1/4 
        font_size = int((x+y/2)*0.03) # draw.text font size
        
        # Draw the face mesh annotations on the image.
        im0f.flags.writeable = True
        im0f = cv2.cvtColor(im0f, cv2.COLOR_RGB2BGR)   

        # Gaze estimation with face mesh   
        if results.multi_face_landmarks:
            for face_landmarks in (results.multi_face_landmarks):
                begin = time.time()
                # Drawing base line(facemesh)
                gaze_model.drawingBaseLine(im0s, face_landmarks)
                gaze_model.total_landmarks.append(face_landmarks.landmark)   
                # Make DataFrames------------------------------------------------------------
                gaze_model.makeDataFrames(face_landmarks, x, y) # iris_df, eyes_df 저장                
                # gaze point line val
                # 눈 좌표 값 방향기준                    
                range_w = int(x * .07) # 좌측부터 2,3번째 그리드의 x좌표 간격에 각각 +,- 값
                # Gaze Point Estimation
                # gaze point line
                eye_list, gaze_line_x, dir_total, box1_x1, box1_x2 = gaze_model.gazeDirection(range_w, x, y)                
                # EAR ratio
                ear, gaze_line_y, box1_y1, box1_y2 = gaze_model.earRatio(face_landmarks, eye_list, y)                
                        
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=args.augment)[0]

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=args.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=args.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Grid line
            gridLine(gaze_model.eyes_df, im0s, range_w, eye_list[0], eye_list[2], x, y)            
            # Gaze line
            gazeLine(im0s, dir_total, ear, x, y)

            iou_key = []
            iou_val = []

            # Process detections
            for i, det in enumerate(pred):  # detections per image                
                if webcam:  # batch_size >= 1
                    p, s, im0s, frame = path[i], '%g: ' % i, im0s, dataset.count
                else:
                    p, s, im0s, frame = path, '',im0s, getattr(dataset, 'frame', 0)

                # gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                # Print results   
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                ##### len(det) 문장 없앰
# --------------------------------------------------------------------------------------------------------- 1 frame 내의 bbox 모두
                box1 = [box1_x1, box1_y1, box1_x2, box1_y2]
                im0s, top_iou_obj, total_fps_cnt, cell_phone_xyxy = object_select_model.IOUandTime(im0s, ear, names, colors, det, box1, x, y)

                # gaze point line
                if ear != 'UP':
                    gazePointLine(im0s, face_landmarks, gaze_line_x, gaze_line_y, x, y, cell_phone_xyxy, top_iou_obj)                
                out.write(im0s)
                # cv2_imshow(im0s)  
                
                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
                print('statement: dir: {}, ear: {}, obj: {}'.format(dir_total, ear, top_iou_obj))
                    
        else: # facemesh 안될 때

            total_fps_cnt += 1/30
            # im0s = noFaceMesh(im0s, total_fps_cnt, x, y)
            im0s = noFaceMesh(im0s, total_fps_cnt, x, y)

            # cv2_imshow(im0s)
            out.write(im0s)

    print('save as',save_path)
    out.release()
    # vid_cap.release()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov7-e6.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='shorts12.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')           

    parser.add_argument('--save-path', type=str, default='run.mp4', help='file/dir/URL/glob')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()    
    with torch.no_grad():
        if args.update:  # update all models (to fix SourceChangeWarning)
            for args.weights in ['best.pt']:
                detect()
                strip_optimizer(args.weights)
        else:
            detect()
