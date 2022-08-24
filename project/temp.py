# 개발용
def detect():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
  
    # variables
    box1_x1,box1_y1,box1_x2,box1_y2 = 0,0,0,0
    top_iou_for10fps = []
    iou_list = []
    top_iou_obj = None
    fps_cnt = 0.0
    total_fps_cnt = 0.0
    top_iou = None
    ear = None
    font=cv2.FONT_HERSHEY_SIMPLEX
    # study object list                
    study_obj = ['tabletcomputer','book','laptop','pen(pencil)']

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)# load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

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
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    first_cnt=0
    
    save_path ='run.mp4'
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
            
        im0f.flags.writeable = False
        im0f = cv2.cvtColor(im0f, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(im0f)
        x = im0f.shape[1] # height
        y = im0f.shape[0] # width
        book_head = y*1/4 
        font_size = int((x+y/2)*0.03) # draw.text font size
        
        # Draw the face mesh annotations on the image.
        im0f.flags.writeable = True
        im0f = cv2.cvtColor(im0f, cv2.COLOR_RGB2BGR)   

        if results.multi_face_landmarks:
            for face_landmarks in (results.multi_face_landmarks):
                begin = time.time()
                # Drawing base line(facemesh)
                # eyes
                mp_drawing.draw_landmarks(
                    image=im0s,
                    landmark_list=face_landmarks,
                    connections=FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                # irises
                mp_drawing.draw_landmarks(
                    image=im0s,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    # mp_face_mesh
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                total_landmarks.append(face_landmarks.landmark)

                # Make DataFrames------------------------------------------------------------
                # iris data frame
                irises=[] # temporary list
                for iris, _ in FACEMESH_IRISES:
                    irises.append(iris)
                irises.sort() # order
                total = [] # to be iris dataframe
                for n,_ in enumerate(irises):
                    n+=1
                    # 좌표 x,y,z값 순서 각 4개씩 (오른쪽눈 < 왼쪽눈) 
                    if n <=len(FACEMESH_LEFT_IRIS):
                        direction = 'right'
                    else:
                        n-=len(FACEMESH_LEFT_IRIS)
                        direction = 'left'
                    now = [_,direction ,face_landmarks.landmark[_].x,face_landmarks.landmark[_].y,face_landmarks.landmark[_].z] # info in this time
                    total.append(now) 
                iris_df = pd.DataFrame(total, columns = ['idx','dir','x','y','z']) # idx: landmark, dir: right/left
                
                # iris / normalized data => resize to origin and to int
                iris_df['x'] = iris_df['x']*x
                iris_df['y'] = iris_df['y']*y
                iris_df['x'] = iris_df['x'].astype('int64')
                iris_df['y'] = iris_df['y'].astype('int64')

                # eyes data frame
                eyes=[] # temporary list
                for eye, _ in FACEMESH_EYES:
                    eyes.append(eye)
                    eyes.append(_)
                eyes = list(set(eyes))
                eyes.sort() # order
                total = [] # to be eyes dataframe
                for n,_ in enumerate(eyes):
                    n+=1
                    # 좌표 x,y,z값 순서 각 16개씩 (오른쪽눈 < 왼쪽눈) 
                    if n <= len(FACEMESH_LEFT_EYE): 
                        direction = 'right'     
                    else:
                        n-=int(len(FACEMESH_LEFT_EYE))
                        direction = 'left'
                    if _ in under:
                        loc = 'under'
                    else:
                        loc = 'up'
                    now = [_,direction ,face_landmarks.landmark[_].x,face_landmarks.landmark[_].y,face_landmarks.landmark[_].z,loc] # info in this time
                    total.append(now)
                eyes_df = pd.DataFrame(total, columns = ['idx','dir','x','y','z','loc']) # idx: landmark, dir: right/left, loc: up/down
                
                # eyes / normalized data => resize to origin and to int
                eyes_df['x'] = eyes_df['x']*x
                eyes_df['y'] = eyes_df['y']*y
                eyes_df['x'] = eyes_df['x'].astype('int64')
                eyes_df['y'] = eyes_df['y'].astype('int64')
                
                # Gaze Point Estimation------------------------------------------------------------
                
                
                # 오른쪽 동공의 각 끝 좌표
                n469_x, n469_y = iris_df[iris_df['idx']==469].x,iris_df[iris_df['idx']==469].y
                n471_x, n471_y = iris_df[iris_df['idx']==471].x,iris_df[iris_df['idx']==471].y
                # 왼쪽 동공의 각 끝 좌표
                n474_x, n474_y = iris_df[iris_df['idx']==474].x,iris_df[iris_df['idx']==474].y
                n476_x, n476_y = iris_df[iris_df['idx']==476].x,iris_df[iris_df['idx']==476].y
                
                # 오른쪽 동공의 중심좌표
                dot_r = ((int(n469_x) + int(n471_x)) / 2, (int(n469_y) + int(n471_y)) / 2)
                # 왼쪽 동공의 중심좌표
                dot_l = ((int(n474_x) + int(n476_x)) / 2, (int(n474_y) + int(n476_y)) / 2)

                # 오른쪽 눈꺼풀의 각 끝 좌표와 길이
                n33 = (eyes_df[eyes_df['idx']==33].x,eyes_df[eyes_df['idx']==33].y)
                n133 = (eyes_df[eyes_df['idx']==133].x,eyes_df[eyes_df['idx']==133].y) 
                # dist_r = math.dist(n33,n133)
                dist_r = distance(eyes_df[eyes_df['idx']==33].iloc[0].x,eyes_df[eyes_df['idx']==33].iloc[0].y,eyes_df[eyes_df['idx']==133].iloc[0].x,eyes_df[eyes_df['idx']==133].iloc[0].y)
                
                # 왼쪽 눈꺼풀의 각 끝 좌표와 길이
                n263 = (eyes_df[eyes_df['idx']==263].x,eyes_df[eyes_df['idx']==263].y)
                n362 = (eyes_df[eyes_df['idx']==362].x,eyes_df[eyes_df['idx']==362].y)
                # dist_l = math.dist(n263,n362)
                dist_l = distance(eyes_df[eyes_df['idx']==263].iloc[0].x,eyes_df[eyes_df['idx']==263].iloc[0].y,eyes_df[eyes_df['idx']==362].iloc[0].x,eyes_df[eyes_df['idx']==362].iloc[0].y)


                # 오른쪽 밑 눈꺼풀
                n145 = (eyes_df[eyes_df['idx']==145].x,eyes_df[eyes_df['idx']==145].y)
                # 왼쪽 밑 눈꺼풀
                n374 = (eyes_df[eyes_df['idx']==374].x,eyes_df[eyes_df['idx']==374].y)
                
                # gaze point line val
                # 눈 좌표 값 방향기준
                
                range_w = int(x*.07) # 좌측부터 2,3번째 그리드의 x좌표 간격에 각각 +,- 값 

                # gaze_point_line --------------------------------------------------
                right_line_x = ((n33[0][1]-range_w)/2)/2
                rightcenter_line_x = ((n33[0][1]-range_w)/2) + ((n33[0][1]-range_w)/2)/2
                center_line_x = (n263[0][17]+range_w - (n33[0][1]-range_w))/2 + (n33[0][1]-range_w)
                leftcenter_line_x = (n263[0][17]+range_w)+(x-(n263[0][17]+range_w))/4
                left_line_x = (n263[0][17]+range_w) + (x-(n263[0][17]+range_w))*3/4
                
                up_line_y = eyes_df[eyes_df['idx']==33].y[1]/2
                middle_line_y = eyes_df[eyes_df['idx']==33].y[1] + (y*.75 -  eyes_df[eyes_df['idx']==33].y[1])/2
                down_line_y = y*.75+y*.125
                
                # 오른쪽 눈 방향 (좌우)
                # r_ratio = round((math.dist(dot_r, n133)/dist_r),5) # if ratio < thres: left
                r_ratio = round(distance((int(n469_x)+int(n471_x))/2,(int(n469_y)+int(n471_y))/2,eyes_df[eyes_df['idx']==133].iloc[0].x,eyes_df[eyes_df['idx']==133].iloc[0].y)/dist_r,5)
                if r_ratio:
                    if r_ratio < thres:
                        dir_r = 'Right'
                    elif r_ratio > thres_:
                        dir_r = 'Left'
                    else:
                        dir_r = 'Center'
                # 왼쪽 눈 방향 (좌우)
                # l_ratio = round((math.dist(dot_l, n263)/dist_l),5) # if ratio < thres: left                
                l_ratio = round(distance((int(n474_x) + int(n476_x)) / 2, (int(n474_y) + int(n476_y)) / 2,eyes_df[eyes_df['idx']==263].iloc[0].x,eyes_df[eyes_df['idx']==263].iloc[0].y)/dist_l,5)
                if l_ratio:
                    if l_ratio < thres:
                        dir_l = 'Right'
                    elif l_ratio > thres_:
                        dir_l = 'Left'
                    else:
                        dir_l = 'Center'

                # 통합 눈 방향 (좌우)
                if dir_r == dir_l:
                    dir_ = dir_r
                    if dir_r == 'Right':
                        gaze_line_x = left_line_x
                        box1_x1, box1_x2 = int((x-(n263[0][17]+range_w))/2+(n263[0][17]+range_w)), x
                    else:
                        gaze_line_x = right_line_x
                        box1_x1, box1_x2 = 0, int((n33[0][1]-range_w)/2)

                elif ((dir_r =='Right') and (dir_l =='Left')) or ((dir_r == 'Left') and (dir_l == 'Right')):
                    dir_ = 'Center' # 양 끝 값일 때, 중앙으로
                    gaze_line_x = center_line_x
                    box1_x1, box1_x2 = n33[0][1]-range_w, n263[0][17]+range_w
                else: # [rightcenter, leftcenter, centerright, centerleft]
                    dir_ = [dir_r,dir_l]
                    if ('Right' in dir_) and ('Center' in dir_):
                        dir_ = 'RightCenter'
                        gaze_line_x = leftcenter_line_x
                        box1_x1, box1_x2 = n263[0][17]+range_w,int((x-(n263[0][17]+range_w))/2+(n263[0][17]+range_w))
                    if ('Left' in dir_) and ('Center' in dir_):
                        dir_ = 'LeftCenter'
                        gaze_line_x = rightcenter_line_x
                        box1_x1, box1_x2 = int((n33[0][1]-range_w)/2),n33[0][1]-range_w
        #                 up_r = iris_df[iris_df['idx']==472]['y'][3] - eyes_df[eyes_df['idx']==145].y[4] # if up<0: up
        #                 up_l = iris_df[iris_df['idx']==477]['y'][7] - eyes_df[eyes_df['idx']==374].y[20] # if up<0: up

                # EAR ratio--------------------------------------------------

                # Right iris(468) z vs Left iris(473) z: higher value is closer camera.
                if face_landmarks.landmark[468].z > face_landmarks.landmark[473].z:
                    using_ear = 'right_ear'
                else:
                    using_ear = 'left_ear'
                
                if using_ear == 'right_ear':
                    # 오른쪽 눈 방향 (상하) : (|161-163|+|157-154|)/2*|133-33|*1/100
                    n161 = (eyes_df[eyes_df['idx']==161].x,eyes_df[eyes_df['idx']==161].y)
                    n163 = (eyes_df[eyes_df['idx']==163].x,eyes_df[eyes_df['idx']==163].y)
                    n154 = (eyes_df[eyes_df['idx']==154].x,eyes_df[eyes_df['idx']==154].y)
                    n157 = (eyes_df[eyes_df['idx']==157].x,eyes_df[eyes_df['idx']==157].y)
                    # right_ear = (abs(math.dist(n161,n163))+abs(math.dist(n157,n154)))/2*abs(math.dist(n133,n33))/1000
                    right_ear = (abs(distance(eyes_df[eyes_df['idx']==161].iloc[0].x,eyes_df[eyes_df['idx']==161].iloc[0].y,eyes_df[eyes_df['idx']==163].iloc[0].x,eyes_df[eyes_df['idx']==163].iloc[0].y))+\
                                abs(distance(eyes_df[eyes_df['idx']==157].iloc[0].x,eyes_df[eyes_df['idx']==157].iloc[0].y,eyes_df[eyes_df['idx']==154].iloc[0].x,eyes_df[eyes_df['idx']==154].iloc[0].y)))/2*\
                                abs(distance(eyes_df[eyes_df['idx']==133].iloc[0].x,eyes_df[eyes_df['idx']==133].iloc[0].y,eyes_df[eyes_df['idx']==33].iloc[0].x,eyes_df[eyes_df['idx']==33].iloc[0].y))/1000
                    using_ear = right_ear

                elif using_ear == 'left_ear':
                    # 왼쪽 눈 방향 (상하) : (|384-381|+|388-390|)/2*|263-362|*1/100
                    n381 = (eyes_df[eyes_df['idx']==381].x,eyes_df[eyes_df['idx']==381].y)
                    n384 = (eyes_df[eyes_df['idx']==384].x,eyes_df[eyes_df['idx']==384].y)
                    n388 = (eyes_df[eyes_df['idx']==388].x,eyes_df[eyes_df['idx']==388].y)
                    n390 = (eyes_df[eyes_df['idx']==390].x,eyes_df[eyes_df['idx']==390].y)
                    # left_ear = (abs(math.dist(n384,n381))+abs(math.dist(n388,n390)))/2*abs(math.dist(n263,n362))/1000
                    left_ear = (abs(distance(eyes_df[eyes_df['idx']==381].iloc[0].x,eyes_df[eyes_df['idx']==381].iloc[0].y,eyes_df[eyes_df['idx']==384].iloc[0].x,eyes_df[eyes_df['idx']==384].iloc[0].y))+\
                                abs(distance(eyes_df[eyes_df['idx']==388].iloc[0].x,eyes_df[eyes_df['idx']==388].iloc[0].y,eyes_df[eyes_df['idx']==390].iloc[0].x,eyes_df[eyes_df['idx']==390].iloc[0].y)))/2*\
                                abs(distance(eyes_df[eyes_df['idx']==263].iloc[0].x,eyes_df[eyes_df['idx']==263].iloc[0].y,eyes_df[eyes_df['idx']==362].iloc[0].x,eyes_df[eyes_df['idx']==362].iloc[0].y))/1000
                    using_ear = left_ear

                if using_ear <= 0.05:
                    ear = 'CLOSE'
                    box1_y1, box1_y2 = int(y*0.75), y # down과 같음
                    gaze_line_y = down_line_y
                elif (using_ear > 0.05) and (using_ear <= thres_ear/2):# thres_ear_ = thres_ear/2
                    ear = 'DOWN'
                    gaze_line_y = down_line_y
                    box1_y1, box1_y2 = int(y*0.75), y
                elif (using_ear > thres_ear/2) and (using_ear < thres_ear): # thres_ear = 0.6
                    ear = 'MIDDLE'
                    gaze_line_y = middle_line_y
                    box1_y1, box1_y2 = eyes_df[eyes_df['idx']==33].y[1], int(y*0.75)
                else:
                    ear = 'UP'
                    gaze_line_y = up_line_y
                    box1_y1, box1_y2 = 0,eyes_df[eyes_df['idx']==33].y[1]
                    
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
                    model(img, augment=opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Grid line--------------------------------------------------
            # out.write(im0s)
            # right 옆 - 왼쪽에서부터 2
            cv2.line(im0s,(n33[0][1]-range_w,0),(n33[0][1]-range_w,y),(255,0,0),1) # n33[0][1]= n33_x, range_w = 50
            # left 옆 - 3
            cv2.line(im0s,(n263[0][17]+range_w,0),(n263[0][17]+range_w,y),(255,0,0),1)

            # right center - 1 
            cv2.line(im0s,(int((n33[0][1]-range_w)/2),0),(int((n33[0][1]-range_w)/2),y),(255,0,0),1)
            # left center - 4
            cv2.line(im0s,(int((x-(n263[0][17]+range_w))/2+(n263[0][17]+range_w)),0),(int((x-(n263[0][17]+range_w))/2+(n263[0][17]+range_w)),y),(255,0,0),1) # n263[0][17]= n263_x
            
            # table
            cv2.line(im0s,(0,int(y*0.75)),(x,int(y*0.75)),(255,0,0),1)
            # eye_line 
            cv2.line(im0s,(0,eyes_df[eyes_df['idx']==33].y[1]),(x,eyes_df[eyes_df['idx']==33].y[1]),(255,0,0),1) 
            #cv2.line(im0s,(0,int(face_landmarks.landmark[10].y*y)),(x,int(face_landmarks.landmark[10].y*y)),(255,0,0),3) # 이마라인선 but, down과 middle의 기준이 애매함, 눈꼬리 기준으로 위아래 나누는게 더 좋을듯
            
            # gaze line(좌우)
            if dir_:
                org=(int(x*0.3),int(y*0.3))
                cv2.putText(im0s,dir_,org,font,.5,(255,0,0),1)
            # gaze line(상하)
            if ear:
                org=(int(x*0.3),int(y*0.4))
                cv2.putText(im0s,ear,org,font,.5,(255,0,0),1)
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
                iou_key = []
                iou_val = []
                iou_xyxy = []
                cell_phone_xyxy = 0
                for  *xyxy, conf, cls in reversed(det):
                    #------------------------------------------------------------------------------1 frame 내의 bbox별 xyxy 박스 그리기
                    xyxy_ = []
                    for _ in xyxy:
                        xyxy_.append(_.item())
                    # print(xyxy_)

                    # 사용물품이 화면 중앙에서 사용 될 때, 탐지 X 보완(ex. cell phone)
                    if names[int(cls.item())] =='book':
                        book_head = xyxy_[1] # 공부X 물품 기준선

                    if (names[int(cls.item())] =='cellphone') and (xyxy_[1] < book_head):
                        cell_phone_xyxy = xyxy # 휴대폰의 위치

                    # print("cls:",names[int(cls.item())])
                    label = f'{names[int(cls)]} {conf:.2f}'
                    # 객체별 bbox 그리기
                    plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=1)

                    # iou
                    if box1_x1 or box1_y1 or box1_x2 or box1_y2:
                        box1 = [box1_x1, box1_y1, box1_x2, box1_y2]
                        box2 = xyxy_
                        iou = IoU(box1,box2)
                        iou_key.append(names[int(cls.item())])
                        iou_val.append(iou)
                        iou_xyxy.append(xyxy_)
                    #-------------------------------------------------------------------------------------
                    # 들여쓰기 주의

                if iou_key or iou_val:
                    top_iou = iou_key[np.argmax(iou_val)] # 1 frame의 가장 높은 값 명사로 저장됨(그리드 기준)
                    
                    if cell_phone_xyxy: # 예외 정보(사용물품 화면중앙에서 사용 될 때의 보완점)
                        top_iou = 'cellphone'
                    
                    top_iou_for10fps.append(top_iou)
                
                im0s = Image.fromarray(im0s) # 한글사용 가능하게 변경(draw.text형식과 같이 움직여야함, cv2line 그릴 때는 array화 시켜야함)
                draw = ImageDraw.Draw(im0s)
                
                if top_iou: # 현재 보는 거
                    org=(int(x*0.1),int(y*0.1))
                    # cv2.putText(im0s,'NOW: '+top_iou, org, font,.5,(255,0,0),1)
                    draw.text(org, "지금 보는 물체:\n"+top_iou, font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255)) # font_size = 20

                # writing => top_iou_obj(10fps동안 빈도수 1등)
                if top_iou_obj: 
                    org=(int(x*0.1),int(y*0.3))
                    # cv2.putText(im0s,top_iou_obj+' in 10FPS',org,font,.5,(255,0,0),1)    
                    draw.text(org, "일정시간동안 보는 물체:\n"+top_iou_obj, font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255))

                # 10개의 프레임 중에 가장 높은 사물
                if top_iou_for10fps:
                    counter_top_iou = Counter(top_iou_for10fps)
                    top_iou_obj = list(counter_top_iou.keys())[(np.argmax(list(counter_top_iou.values())))] # 명사로 저장됨
                    # print('top_iou_for10fps :',top_iou_for10fps)
                    # print('top_iou_obj:',top_iou_obj)
                # 최근 10개의 프레임
                if len(top_iou_for10fps) == 10: 
                    top_iou_for10fps = top_iou_for10fps[1:]                      
                
                if top_iou_obj in study_obj:
                    fps_cnt += 1/30 # 순공시간, 1단위: 1초 
                    if ear == 'CLOSE': # 졸음 시간
                        org = (int(x*0.35),int(y*0.45))
                        draw.text(org, "혹시 졸고 계신가요?(시간측정X)", font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255))
                        fps_cnt -= 1/30
                    
                if fps_cnt: # 현재 시간
                    org=(int(x*0.7),int(y*0.3))
                    # cv2.putText(im0s,'now: {}:{}:{:.3f}'.format(int(fps_cnt//60),int(fps_cnt//1),fps_cnt%1), org, font,.5,(255,0,0),1)
                    draw.text(org, "순공부시간\n{}:{}:{:.3f}".format(int(fps_cnt//60),int(fps_cnt//1),fps_cnt%1), font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255))              
                            
                total_fps_cnt += 1/30 # 전체시간
                if total_fps_cnt: # 전체시간
                    org=(int(x*0.7),int(y*0.1))
                    # cv2.putText(im0s,'total: {}:{}:{:.3f}'.format(int(total_fps_cnt//60),int(total_fps_cnt//1),total_fps_cnt%1), org, font,.5,(255,0,0),1)                    
                    draw.text(org, "전체시간\n{}:{}:{:.3f}".format(int(total_fps_cnt//60),int(total_fps_cnt//1),total_fps_cnt%1), font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255))              
                
                im0s = np.array(im0s) # 한글사용 불가능하게 변경
                    
                # gaze point line
                if ear != 'UP':
                    # if (xyxy_[1] < book_head) and (names[int(cls.item())] =='cellphone'):
                    # if (iou_xyxy[np.argmax(iou_val)][1] < book_head):
                    
                    if cell_phone_xyxy: # 휴대폰 사용시
                        if top_iou_obj =='cellphone':
                            cv2.line(im0s,(int(face_landmarks.landmark[468].x * x),int(face_landmarks.landmark[468].y * y)),
                                (int((cell_phone_xyxy[0] + cell_phone_xyxy[2]) / 2 - x * .02),int((cell_phone_xyxy[1] + cell_phone_xyxy[3]) / 2))
                                ,(255,0,0),2)
                            cv2.line(im0s,(int(face_landmarks.landmark[473].x * x),int(face_landmarks.landmark[473].y * y)),
                                (int((cell_phone_xyxy[0] + cell_phone_xyxy[2]) / 2 + x * .02),int((cell_phone_xyxy[1] + cell_phone_xyxy[3]) / 2))
                                ,(255,0,0),2)  
                        
                    else: # 시선의 그리드로 line
                        cv2.line(im0s,(int(face_landmarks.landmark[468].x*x),int(face_landmarks.landmark[468].y*y)),(int(gaze_line_x-x*.07), int(gaze_line_y)),(102,204,0),2) 
                        cv2.line(im0s,(int(face_landmarks.landmark[473].x*x),int(face_landmarks.landmark[473].y*y)),(int(gaze_line_x+x*.07), int(gaze_line_y)),(102,204,0),2)

                    # im0s = Image.fromarray(im0s) # 한글사용 가능하게 변경(draw.text형식과 같이 움직여야함, cv2line 그릴 때는 array화 시켜야함)
                    # draw = ImageDraw.Draw(im0s)
                    # if fps_cnt: # 현재 시간
                    #     org=(int(x*0.7),int(y*0.3))
                    #     # cv2.putText(im0s,'now: {}:{}:{:.3f}'.format(int(fps_cnt//60),int(fps_cnt//1),fps_cnt%1), org, font,.5,(255,0,0),1)
                    #     draw.text(org, "순공부시간\n{}:{}:{:.3f}".format(int(fps_cnt//60),int(fps_cnt//1),fps_cnt%1), font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255))              
                            
                    # total_fps_cnt += 1/30 # 전체시간
                    # if total_fps_cnt: # 전체시간
                    #     org=(int(x*0.7),int(y*0.1))
                    #     # cv2.putText(im0s,'total: {}:{}:{:.3f}'.format(int(total_fps_cnt//60),int(total_fps_cnt//1),total_fps_cnt%1), org, font,.5,(255,0,0),1)                    
                    #     draw.text(org, "전체시간\n{}:{}:{:.3f}".format(int(total_fps_cnt//60),int(total_fps_cnt//1),total_fps_cnt%1), font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255))              
                    
                    # im0s = np.array(im0s) # 한글사용 불가능하게 변경                        
                    
                out.write(im0s)
                # cv2.imshow('',im0s)
                
                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
                print('statement: dir: {}, ear: {}, obj: {}'.format(dir_, ear, top_iou_obj))
                
                
        else: # facemesh 안될 때
        
            total_fps_cnt += 1/30
            
            im0s = Image.fromarray(im0s) # 한글사용 가능하게 변경
            draw = ImageDraw.Draw(im0s)
                    
            if total_fps_cnt:
                org=(int(x*0.7),int(y*0.1))
                # cv2.putText(im0s,'total: {}:{}:{:.3f}'.format(int(total_fps_cnt//60),int(total_fps_cnt//1),total_fps_cnt%1), org, font,.5,(255,0,0),1) 
                draw.text(org, "전체시간\n{}:{}:{:.3f}".format(int(total_fps_cnt//60),int(total_fps_cnt//1),total_fps_cnt%1), font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255))   
            # WARNING
            org=(int(x*0.2),int(y*0.35))
            # cv2.putText(im0s,"It Doesn't Work. Please Follow the Instructions.", org, font,.5,(255,0,0),1)
            draw.text(org, "눈이 잘 안보여요. \n졸고 계신거 아니죠?", font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255))
            # table text
            org=(int(x*0.6),int(y*0.65))
            # cv2.putText(im0s,'SET TABLE, HERE', org, font,.5,(255,0,0),1)
            draw.text(org,'아래에 책상선을 맞춰주세요.', font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255))
            # head circle text
            org=(int(x*0.45),int(y*0.2))
            # cv2.putText(im0s,'SET HEAD', org, font,.5,(255,0,0),1)
            draw.text(org, "여기는 머리", font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255))
            im0s = np.array(im0s) # 한글사용 불가능하게 변경
                    
            # table line                
            cv2.line(im0s,(0,int(y*0.75)),(x,int(y*0.75)),(255,255,255),1)
            # head circle
            cv2.circle(im0s,(int(x*0.5), int(y*0.3)), int(x*0.08), (255, 255, 255), 1, cv2.LINE_AA)
            
            out.write(im0s)
            # cv2_imshow(im0s)   

    out.release()
    # vid_cap.release()