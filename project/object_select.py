import numpy as np
from collections import Counter
from PIL import ImageFont, ImageDraw, Image

from utils.plots import plot_one_box
from utils2 import *

class ObjectSelect:
    def __init__(self, args):
        #self.box1_x1, self.box1_y1, self.box1_x2, self.box1_y2 = 0,0,0,0
        self.top_iou_for10fps = []
        #self.iou_list = []
        self.top_iou_obj = None
        self.fps_cnt = 0.0
        self.total_fps_cnt = 0.0
        self.top_iou = None
        #self.ear = None

    def IOUandTime(self, image, ear,  names, colors, det, box1, x, y):
        
        # box1_x1, box1_y1, box1_x2, box1_y2 = box1[0:]
        iou_key = []
        iou_val = []
        iou_xyxy = []
        cell_phone_xyxy = 0
        study_obj = ['tabletcomputer','book','laptop','pen(pencil)']
        book_head = y*1/4 

        # font=cv2.FONT_HERSHEY_SIMPLEX
        font_size = int((x+y/2)*0.03) # draw.text font size


        for  *xyxy, conf, cls in reversed(det):
            
            # 1 frame 내의 bbox별 xyxy 박스 그리기
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
            plot_one_box(xyxy, image, label=label, color=colors[int(cls)], line_thickness=1)

            # iou
            if box1:
                box2 = xyxy_
                iou = IoU(box1,box2)
                iou_key.append(names[int(cls.item())])
                iou_val.append(iou)
                iou_xyxy.append(xyxy_)

        if iou_key or iou_val:
            self.top_iou = iou_key[np.argmax(iou_val)] # 1 frame의 가장 높은 값 명사로 저장됨(그리드 기준)
            
            if cell_phone_xyxy: # 예외 정보(사용물품 화면중앙에서 사용 될 때의 보완점)
                self.top_iou = 'cellphone'
            
            self.top_iou_for10fps.append(self.top_iou)
        
        image = Image.fromarray(image) # 한글사용 가능하게 변경(draw.text형식과 같이 움직여야함, cv2line 그릴 때는 array화 시켜야함)
        draw = ImageDraw.Draw(image)
        
        if self.top_iou: # 현재 보는 거
            org=(int(x*0.1),int(y*0.1))
            # cv2.putText(im0s,'NOW: '+top_iou, org, font,.5,(255,0,0),1)
            draw.text(org, "지금 보는 물체:\n"+self.top_iou, font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255)) # font_size = 20

        # writing => top_iou_obj(10fps동안 빈도수 1등)
        if self.top_iou_obj: 
            org=(int(x*0.1),int(y*0.3))
            # cv2.putText(im0s,top_iou_obj+' in 10FPS',org,font,.5,(255,0,0),1)    
            draw.text(org, "일정시간동안 보는 물체:\n"+self.top_iou_obj, font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255))

        # 10개의 프레임 중에 가장 높은 사물
        if self.top_iou_for10fps:
            counter_top_iou = Counter(self.top_iou_for10fps)
            self.top_iou_obj = list(counter_top_iou.keys())[(np.argmax(list(counter_top_iou.values())))] # 명사로 저장됨
            # print('top_iou_for10fps :',top_iou_for10fps)
            # print('top_iou_obj:',top_iou_obj)
        # 최근 10개의 프레임
        if len(self.top_iou_for10fps) == 10: 
            self.top_iou_for10fps = self.top_iou_for10fps[1:]                      
        
        if self.top_iou_obj in study_obj:
            self.fps_cnt += 1/30 # 순공시간, 1단위: 1초 
            if ear == 'CLOSE': # 졸음 시간
                org = (int(x*0.35),int(y*0.45))
                draw.text(org, "혹시 졸고 계신가요?(시간측정X)", font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255))
                self.fps_cnt -= 1/30
            
        if self.fps_cnt: # 현재 시간
            org=(int(x*0.7),int(y*0.3))
            # cv2.putText(im0s,'now: {}:{}:{:.3f}'.format(int(fps_cnt//60),int(fps_cnt//1),fps_cnt%1), org, font,.5,(255,0,0),1)
            draw.text(org, "순공부시간\n{}:{}:{:.3f}".format(int(self.fps_cnt//60),int(self.fps_cnt//1),self.fps_cnt%1), font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255))              
                    
        self.total_fps_cnt += 1/30 # 전체시간
        if self.total_fps_cnt: # 전체시간
            org=(int(x*0.7),int(y*0.1))
            # cv2.putText(im0s,'total: {}:{}:{:.3f}'.format(int(total_fps_cnt//60),int(total_fps_cnt//1),total_fps_cnt%1), org, font,.5,(255,0,0),1)                    
            draw.text(org, "전체시간\n{}:{}:{:.3f}".format(int(self.total_fps_cnt//60),int(self.total_fps_cnt//1),self.total_fps_cnt%1), font=ImageFont.truetype("./Nanum_hand_seongsil.ttf", font_size), fill=(255,255,255))              
        
        image = np.array(image) # 한글사용 불가능하게 변경
        
        return image, self.top_iou_obj, self.total_fps_cnt, cell_phone_xyxy       