import cv2
import math
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# landmark 정의
FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)])
FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])
FACEMESH_CONTOURS = frozenset().union(*[FACEMESH_LEFT_EYE, FACEMESH_RIGHT_EYE])
FACEMESH_RIGHT_IRIS = frozenset([(469, 470), (470, 471), (471, 472),(472, 469)])
FACEMESH_LEFT_IRIS = frozenset([(474, 475), (475, 476), (476, 477),(477, 474)])
FACEMESH_IRISES = frozenset().union(*[FACEMESH_LEFT_IRIS, FACEMESH_RIGHT_IRIS])
FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                       (374, 380), (380, 381), (381, 382), (382, 362),
                       (263, 466), (466, 388), (388, 387), (387, 386),
                       (386, 385), (385, 384), (384, 398), (398, 362)])
FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])
FACEMESH_EYES = frozenset().union(*[FACEMESH_LEFT_EYE, FACEMESH_RIGHT_EYE])
right_under = [7,33,133,144,145,153,154,155,163]
left_under = [249,263,362,373,374,380,381,382,390]
under = frozenset().union(right_under,left_under) # under에는 양끝 눈꺼풀도 포함되어 있음

def distance(x1, y1, x2, y2):
    result = math.sqrt( math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    return result

# right_ear = (abs(math.dist(n161,n163))+abs(math.dist(n157,n154)))/2*abs(math.dist(n133,n33))/1000
def earCal(n1, n2, n3, n4, n5, n6):
    ear = (abs(distance(n1[0].iloc[0], n1[1].iloc[0], n2[0].iloc[0], n2[1].iloc[0]))+\
           abs(distance(n3[0].iloc[0], n3[1].iloc[0], n4[0].iloc[0], n4[1].iloc[0])))/2*\
           abs(distance(n5[0].iloc[0], n5[1].iloc[0], n6[0].iloc[0], n6[1].iloc[0]))/1000
    return ear 

def makeDataFrame(face_landmarks, FACEMESH, length, flag):

    temps = [] # temporary list        
    for i, _ in FACEMESH:
        temps.append(i)      
        if flag == 1:  
            temps.append(_)
    if flag == 1:
        temps = list(set(temps))        
    temps.sort() # order
    
    total = [] # to be dataframe        
    for n,_ in enumerate(temps):
        n += 1
        # iris : 좌표 x,y,z값 순서 각 4개씩 (오른쪽눈 < 왼쪽눈) 
        # eye : 좌표 x,y,z값 순서 각 16개씩 (오른쪽눈 < 왼쪽눈) 
        if n <= length: 
            direction = 'right'     
        else:
            n -= int(length)
            direction = 'left'
        
        if flag == 1:
            if _ in under:
                loc = 'under'
            else:
                loc = 'up'
        
        if flag == 1:
            now = [_,direction ,face_landmarks.landmark[_].x,face_landmarks.landmark[_].y,face_landmarks.landmark[_].z,loc] # info in this time
        else:
            now = [_,direction ,face_landmarks.landmark[_].x,face_landmarks.landmark[_].y,face_landmarks.landmark[_].z] # info in this time
        total.append(now)

    return total

def resizeToOrigin(df, x, y):
    df['x'] = df['x']*x
    df['y'] = df['y']*y
    df['x'] = df['x'].astype('int64')
    df['y'] = df['y'].astype('int64')
    return df

def gridLine(eyes_df, image, range_w, n33, n263, x, y):

    # right 옆 - 왼쪽에서부터 2
    cv2.line(image,(n33[0][1]-range_w,0),(n33[0][1]-range_w,y),(255,0,0),3) # n33[0][1]= n33_x, range_w = 50
     # left 옆 - 3
    cv2.line(image,(n263[0][17]+range_w,0),(n263[0][17]+range_w,y),(255,0,0),3)

    # right center - 1
    cv2.line(image,(int((n33[0][1]-range_w)/2),0),(int((n33[0][1]-range_w)/2),y),(255,0,0),3)
    # right center - 4
    cv2.line(image,(int((x-(n263[0][17]+range_w))/2+(n263[0][17]+range_w)),0),(int((x-(n263[0][17]+range_w))/2+(n263[0][17]+range_w)),y),(255,0,0),3) # n263[0][17]= n263_x
    
    # table
    cv2.line(image,(0,int(y*0.75)),(x,int(y*0.75)),(255,0,0),1) # 책상선
    # eye_line
    cv2.line(image,(0,eyes_df[eyes_df['idx']==33].y[1]),(x,eyes_df[eyes_df['idx']==33].y[1]),(255,0,0),1) # 오른쪽 바깥 눈꼬리 기준

    #cv2.line(image,(0,int(face_landmarks.landmark[10].y*y)),(x,int(face_landmarks.landmark[10].y*y)),(255,0,0),3) # 이마라인선 but, down과 middle의 기준이 애매함, 눈꼬리 기준으로 위아래 나누는게 더 좋을듯

def gazePointLine_lite(image, face_landmarks, ear, gaze_line_x, gaze_line_y, x, y):
    if ear != 'UP':
        cv2.line(image,(int(face_landmarks.landmark[468].x*x),int(face_landmarks.landmark[468].y*y)),(int(gaze_line_x-x*.07), int(gaze_line_y)),(255,0,0),2) 
        cv2.line(image,(int(face_landmarks.landmark[473].x*x),int(face_landmarks.landmark[473].y*y)),(int(gaze_line_x+x*.07), int(gaze_line_y)),(255,0,0),2)

def gazePointLine(image, face_landmarks, gaze_line_x, gaze_line_y, x, y, cell_phone_xyxy, top_iou_obj):
    if cell_phone_xyxy: # 휴대폰 사용시
        if top_iou_obj =='cellphone':
            cv2.line(image,(int(face_landmarks.landmark[468].x * x),int(face_landmarks.landmark[468].y * y)),
                (int((cell_phone_xyxy[0] + cell_phone_xyxy[2]) / 2 - x * .02),int((cell_phone_xyxy[1] + cell_phone_xyxy[3]) / 2))
                ,(255,0,0),2)
            cv2.line(image,(int(face_landmarks.landmark[473].x * x),int(face_landmarks.landmark[473].y * y)),
                (int((cell_phone_xyxy[0] + cell_phone_xyxy[2]) / 2 + x * .02),int((cell_phone_xyxy[1] + cell_phone_xyxy[3]) / 2))
                ,(255,0,0),2)          
    else: # 시선의 그리드로 line
        cv2.line(image,(int(face_landmarks.landmark[468].x*x),int(face_landmarks.landmark[468].y*y)),(int(gaze_line_x-x*.07), int(gaze_line_y)),(102,204,0),2) 
        cv2.line(image,(int(face_landmarks.landmark[473].x*x),int(face_landmarks.landmark[473].y*y)),(int(gaze_line_x+x*.07), int(gaze_line_y)),(102,204,0),2)                      

def gazeLine(image, dir_total, ear, x, y):
    font=cv2.FONT_HERSHEY_SIMPLEX
    # gaze line(좌우)
    if dir_total:
        org=(int(x*0.3),int(y*0.3))
        cv2.putText(image,dir_total,org,font,.5,(255,0,0),1)
    # gaze line(상하)
    if ear:
        org=(int(x*0.3),int(y*0.4))
        cv2.putText(image,ear,org,font,.5,(255,0,0),1)    

def putText(image, dir_total, ear, x, y):
    
    if dir_total:
        org=(int(x*0.3),int(y*0.3))
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image,dir_total,org,font,.5,(255,0,0),1)
        # size, BaseLine=cv2.getTextSize(dir_,font,1,2)
    if ear:
        org=(int(x*0.3),int(y*0.4))
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image,ear,org,font,.5,(255,0,0),1)
        # size, BaseLine=cv2.getTextSize(ear,font,1,2)

# box1 = box1_x1,box1_y1,box1_x2,box1_y2
# box2 = xyxy_
def IoU(box1, box2): # box1이 그리드, box2가 객체 박스
    global box1_x1,box1_y1,box1_x2,box1_y2
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    self_iou = inter / box2_area
    return self_iou

def noFaceMesh(image, total_fps_cnt, x, y):


    # font=cv2.FONT_HERSHEY_SIMPLEX
    font_size = int((x+y/2)*0.03) # draw.text font size


    image = Image.fromarray(image) # 한글사용 가능하게 변경
    draw = ImageDraw.Draw(image)
            
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

    return image

    # image = Image.fromarray(image) # 한글사용 가능하게 변경
    # draw = ImageDraw.Draw(image)



    # if total_fps_cnt:
    #     org=(int(x*0.7),int(y*0.1))
    #     # cv2.putText(im0s,'total: {}:{}:{:.3f}'.format(int(total_fps_cnt//60),int(total_fps_cnt//1),total_fps_cnt%1), org, font,.5,(255,0,0),1) 
    #     draw.text(org, "전체시간\n{}:{}:{:.3f}".format(int(total_fps_cnt//60),int(total_fps_cnt//1),total_fps_cnt%1), font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255))   
    # # WARNING
    # org=(int(x*0.2),int(y*0.3))
    # # cv2.putText(im0s,"It Doesn't Work. Please Follow the Instructions.", org, font,.5,(255,0,0),1)
    # draw.text(org, "눈이 잘 안보여요. \n졸고 계신게 아니라면, 안내선에 따라주세요!", font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255))
    # # table text
    # org=(int(x*0.5),int(y*0.7))
    # # cv2.putText(im0s,'SET TABLE, HERE', org, font,.5,(255,0,0),1)
    # draw.text(org,'아래에 책상선을 맞춰주세요.', font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255))
    # # head circle text
    # org=(int(x*0.46),int(y*0.2))
    # # cv2.putText(im0s,'SET HEAD', org, font,.5,(255,0,0),1)
    # draw.text(org, "여기는 머리", font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255))
    # image = np.array(image) # 한글사용 불가능하게 변경
            
    # # table line                
    # cv2.line(image,(0,int(y*0.75)),(x,int(y*0.75)),(255,0,0),1)
    # # head circle
    # cv2.circle(image,(int(x*0.5), int(y*0.3)), int(x*0.08), (255, 0, 0), 1, cv2.LINE_AA) 

    # out.write(image)
    # # cv2_imshow(im0s)

    # return image

# def box:
#     # 통합 눈 방향 (좌우)
#     if dir_r == dir_l:
#         dir_ = dir_r
#         if dir_r == 'Right':
#             box1_x1, box1_x2 = int((x-(n263[0][17]+range_w))/2+(n263[0][17]+range_w)), x
#         else:
#             box1_x1, box1_x2 = 0, int((n33[0][1]-range_w)/2)

#     elif ((dir_r =='Right') and (dir_l =='Left')) or ((dir_r == 'Left') and (dir_l == 'Right')):
#         dir_ = 'Center' # 양 끝 값일 때, 중앙으로
#         box1_x1, box1_x2 = n33[0][1]-range_w, n263[0][17]+range_w
#     else: # [rightcenter, leftcenter, centerright, centerleft]
#         dir_ = [dir_r,dir_l]
#         if ('Right' in dir_) and ('Center' in dir_):
#             dir_ = 'RightCenter'
#             box1_x1, box1_x2 = n263[0][17]+range_w,int((x-(n263[0][17]+range_w))/2+(n263[0][17]+range_w))
#         if ('Left' in dir_) and ('Center' in dir_):
#             dir_ = 'LeftCenter'
#             box1_x1, box1_x2 = int((n33[0][1]-range_w)/2),n33[0][1]-range_w