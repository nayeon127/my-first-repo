#from google.colab.patches import cv2_imshow
import cv2
import mediapipe as mp
import pandas as pd
import time
from utils2 import * 

import argparse
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve() # FILE은 현재 파일의 절대경로 (예를 들어, C:\Users\...)
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# landmark 정의는 util에 있습니다.

class GazeEstimation:
    
    def __init__(self, args):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        # For webcam input:
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        self.path = args.source
        self.save_path = args.save_path
        # variable
        self.total_landmarks = []
        self.time_list = []
        self.ok_flag = 1
        self.num = 1
        self.thres = 0.45 # thres < 0.5 (select in 0.40 ~ 0.47)  => e.g. [thres|----|(1-thres)]  
        self.thres_ = 1 - self.thres
        self.thres_ear = 0.7 # thres_ear >= 0.5 => up
        # self.range_w = 50 # 좌측부터 2,3번째 그리드의 x좌표 간격에 각각 +,- 값

        pd.set_option('mode.chained_assignment', None)        

    def faceMeshLoad(self, image):
        face_mesh = self.mp_face_mesh.FaceMesh( max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        return face_mesh.process(image)

    def drawingBaseLine(self, image, face_landmarks):
        # eyes
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
        )
        # irises
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_IRISES,
            # mp_face_mesh
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        )

    def makeDataFrames(self, face_landmarks, x, y):
        # iris data frame
        total_iris = makeDataFrame(face_landmarks, FACEMESH_IRISES, len(FACEMESH_LEFT_IRIS), flag=0)
        self.iris_df = pd.DataFrame(total_iris, columns = ['idx','dir','x','y','z']) # idx: landmark, dir: right/left        
        # iris / normalized data => resize to origin and to int
        resizeToOrigin(self.iris_df, x, y)
        # eyes data frame
        total_eye = makeDataFrame(face_landmarks, FACEMESH_EYES, len(FACEMESH_LEFT_EYE), flag=1)
        self.eyes_df = pd.DataFrame(total_eye, columns = ['idx','dir','x','y','z','loc']) # idx: landmark, dir: right/left, loc: up/down        
        # eyes / normalized data => resize to origin and to int
        resizeToOrigin(self.eyes_df, x, y)
        
    def gazeDirection(self, range_w, x, y):        
        
        # gaze Point --------------------------------------------------------------------------
        iris_df = self.iris_df
        eyes_df = self.eyes_df
        eye_list = []
        
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
        eye_r1 = (eyes_df[eyes_df['idx']==33].x,eyes_df[eyes_df['idx']==33].y)   # n33
        eye_r2 = (eyes_df[eyes_df['idx']==133].x,eyes_df[eyes_df['idx']==133].y) # n133
        dist_r = distance(eye_r1[0].iloc[0], eye_r1[1].iloc[0], eye_r2[0].iloc[0], eye_r2[1].iloc[0])
        # 왼쪽 눈꺼풀의 각 끝 좌표와 길이
        eye_l1 = (eyes_df[eyes_df['idx']==263].x,eyes_df[eyes_df['idx']==263].y) # n263
        eye_l2 = (eyes_df[eyes_df['idx']==362].x,eyes_df[eyes_df['idx']==362].y) # n362
        dist_l = distance(eye_l1[0].iloc[0], eye_l1[1].iloc[0], eye_l2[0].iloc[0], eye_l2[1].iloc[0])

        eye_list = [eye_r1, eye_r2, eye_l1, eye_l2]
        
        # gaze Point Line --------------------------------------------------------------------------
        dir_r = None        # 오른쪽 눈 방향
        dir_l = None        # 왼쪽 눈 방향
        dir_total = None    # 통합 눈 방향

        # 눈 좌표 값 방향기준
        right_line_x = ((eye_r1[0][1]-range_w)/2)/2
        rightcenter_line_x = ((eye_r1[0][1]-range_w)/2) + ((eye_r1[0][1]-range_w)/2)/2
        center_line_x = (eye_l1[0][17]+range_w - (eye_r1[0][1]-range_w))/2 + (eye_r1[0][1]-range_w)
        leftcenter_line_x = (eye_l1[0][17]+range_w)+(x-(eye_l1[0][17]+range_w))/4
        left_line_x = (eye_l1[0][17]+range_w) + (x-(eye_l1[0][17]+range_w))*3/4
        
        # 오른쪽 눈 방향 (좌우)
        r_ratio = round(distance(dot_r[0], dot_r[1], eye_r2[0].iloc[0], eye_r2[1].iloc[0])/dist_r,5) # if ratio < thres: left        
        if r_ratio:
            if r_ratio < self.thres:
                dir_r = 'Right'
            elif r_ratio > self.thres_:
                dir_r = 'Left'
            else:
                dir_r = 'Center'
        
        # 왼쪽 눈 방향 (좌우)
        l_ratio = round(distance(dot_l[0], dot_l[1], eye_l1[0].iloc[0], eye_l1[1].iloc[0])/dist_l,5) # if ratio < thres: left        
        if l_ratio:
            if l_ratio < self.thres:
                dir_l = 'Right'
            elif l_ratio > self.thres_:
                dir_l = 'Left'
            else:
                dir_l = 'Center'   
        
        # 통합 눈 방향 (좌우)
        # 양쪽 눈의 방향이 같은 경우 그 방향을 본다고 판단한다.
        if dir_r == dir_l:
            dir_total = dir_r
            if dir_r == 'Right':
                gaze_line_x = left_line_x
                box1_x1, box1_x2 = int((x-(eye_l1[0][17]+range_w))/2+(eye_l1[0][17]+range_w)), x
            else:
                gaze_line_x = right_line_x
                box1_x1, box1_x2 = 0, int((eye_r1[0][1]-range_w)/2)        
        
        # 양쪽 눈의 방향이 전혀 다른 경우 중앙을 본다고 판단한다.
        elif ((dir_r =='Right') and (dir_l =='Left')) or ((dir_r == 'Left') and (dir_l == 'Right')):
            dir_total = 'Center' # 양 끝 값일 때, 중앙으로
            gaze_line_x = center_line_x
            box1_x1, box1_x2 = eye_r1[0][1]-range_w, eye_l1[0][17]+range_w
        
        # 양쪽 눈의 방향이 약간 다른 경우 그 중간을 본다고 판단한다.
        else: # [rightcenter, leftcenter, centerright, centerleft]
            dir_total = [dir_r,dir_l]
            if ('Right' in dir_total) and ('Center' in dir_total):
                dir_total = 'RightCenter'
                gaze_line_x = leftcenter_line_x
                box1_x1, box1_x2 = eye_l1[0][17]+range_w,int((x-(eye_l1[0][17]+range_w))/2+(eye_l1[0][17]+range_w))
            if ('Left' in dir_total) and ('Center' in dir_total):
                dir_total = 'LeftCenter'
                gaze_line_x = rightcenter_line_x
                box1_x1, box1_x2 = int((eye_r1[0][1]-range_w)/2),eye_r1[0][1]-range_w

        return eye_list, gaze_line_x, dir_total, box1_x1, box1_x2

    def earRatio(self, face_landmarks, eye_list, y):
        
        # EAR Point --------------------------------------------------------------------------
        eyes_df = self.eyes_df
        eye_r1, eye_r2, eye_l1, eye_l2 = eye_list[0:]

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
            right_ear = earCal(n161, n163, n157, n154, eye_r2, eye_r1)        
            # right_ear = (abs(distance(n161[0].iloc[0], n161[1].iloc[0], n163[0].iloc[0], n163[1].iloc[0]))+\
            #             abs(distance(n157[0].iloc[0], n157[1].iloc[0], n154[0].iloc[0], n154[1].iloc[0])))/2*\
            #             abs(distance(eye_r2[0].iloc[0], eye_r2[1].iloc[0], eye_r2[0].iloc[0], eye_r1[1].iloc[0]))/1000        
            using_ear = right_ear

        elif using_ear == 'left_ear':
            # 왼쪽 눈 방향 (상하) : (|384-381|+|388-390|)/2*|263-362|*1/100
            n381 = (eyes_df[eyes_df['idx']==381].x,eyes_df[eyes_df['idx']==381].y)
            n384 = (eyes_df[eyes_df['idx']==384].x,eyes_df[eyes_df['idx']==384].y)
            n388 = (eyes_df[eyes_df['idx']==388].x,eyes_df[eyes_df['idx']==388].y)
            n390 = (eyes_df[eyes_df['idx']==390].x,eyes_df[eyes_df['idx']==390].y)
            left_ear = earCal(n384, n381, n388, n390, eye_l1, eye_l2)        
            # left_ear = (abs(distance(n384[0].iloc[0], n384[1].iloc[0], n381[0].iloc[0], n381[1].iloc[0]))+\
            #             abs(distance(n388[0].iloc[0], n388[1].iloc[0], n390[0].iloc[0], n390[1].iloc[0])))/2*\
            #             abs(distance(eye_l1[0].iloc[0], eye_l1[1].iloc[0], eye_l2[0].iloc[0], eye_l2[1].iloc[0]))/1000
            using_ear = left_ear

        # EAR Estimation --------------------------------------------------------------------------
        up_line_y = eyes_df[eyes_df['idx']==33].y[1]/2
        middle_line_y = eyes_df[eyes_df['idx']==33].y[1] + (y*.75 - eyes_df[eyes_df['idx']==33].y[1])/2
        down_line_y = y*.75+y*.125

        if using_ear <= 0.15:
            ear = 'CLOSE'
            gaze_line_y = down_line_y
            box1_y1, box1_y2 = int(y*0.75), y # down과 같음
        elif (using_ear > 0.15) and (using_ear <= self.thres_ear/2):# thres_ear_ = thres_ear/2
            ear = 'DOWN'
            gaze_line_y = down_line_y
            box1_y1, box1_y2 = int(y*0.75), y
        elif (using_ear > 0.4) and (using_ear < self.thres_ear): # thres_ear = 0.7
            ear = 'MIDDLE'
            gaze_line_y = middle_line_y
            box1_y1, box1_y2 = eyes_df[eyes_df['idx']==33].y[1], int(y*0.75)
        else:
            ear = 'UP'
            gaze_line_y = up_line_y
            box1_y1, box1_y2 = 0,eyes_df[eyes_df['idx']==33].y[1]        
        
        return ear, gaze_line_y, box1_y1, box1_y2
    
    def timeCalculation(self, begin):
        end = time.time()
        t = end - begin # 현재 frame 시간 값
        
#                 if obj in positive_list: # 공부하는 시간은 frame시간 더함
#                     study_time += t # 순공시간
#                 else:
#                     continue    
#                 now = [t,obj] # 현재 행
#                 time_list.append(now) # sum 했을 때 => 전체 시간 값          


    def run(self):

        pd.set_option('mode.chained_assignment', None)

        dir_total = None

        path = str(self.path)
        save_path = str(self.save_path)
        print(path, save_path)
        print(type(path), type(save_path))
        print(ROOT/ 'run.mp4')

        # default: None, * 휴대폰 촬영 영상 -> 이후 코드에서 180도 회전 주석 확인
        # example video: shorts12: , short16:측정불가, shorts17: 2명이상, shorts26: 안구운동 

        if path:
            cap = cv2.VideoCapture(path)
        else:
            cap = cv2.VideoCapture(0)
        if cap:  
            w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        while cap.isOpened() and self.ok_flag == 1:
            
            success, image = cap.read()
            self.num += 1 # frame 개수
            # image=image[::-1] # 주의, 폰으로 촬영시 180도 뒤집히는 현상
            if not success:
                print("Ignoring empty camera frame.")
                break # If loading a video, use 'break' instead of 'continue'.

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.faceMeshLoad(image)
            x = image.shape[1] # height
            y = image.shape[0] # width

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.multi_face_landmarks:
                
                for face_landmarks in (results.multi_face_landmarks):
                    
                    begin = time.time()
                    
                    # Drawing base line (facemesh)
                    self.drawingBaseLine(image, face_landmarks)
                    self.total_landmarks.append(face_landmarks.landmark)
                    
                    # Make DataFrames
                    self.makeDataFrames(face_landmarks, x, y) # iris_df, eyes_df 저장
                    
                    # gaze point line val
                    # 눈 좌표 값 방향기준                    
                    range_w = int(x * .07) # 좌측부터 2,3번째 그리드의 x좌표 간격에 각각 +,- 값
                    # Gaze Point Estimation
                    # gaze point line
                    eye_list, gaze_line_x, dir_total, box1_x1, box1_x2 = self.gazeDirection(range_w, x, y)
                                        
                    # EAR ratio
                    ear, gaze_line_y, box1_y1, box1_y2 = self.earRatio(face_landmarks, eye_list, y)
                    
                    # 화면에 그리기 -> 3개의 method는 util에 있습니다.
                    # Grid line
                    gridLine(self.eyes_df, image, range_w, eye_list[0], eye_list[2], x, y)
                    # gaze point line
                    gazePointLine_lite(image, face_landmarks, ear, gaze_line_x, gaze_line_y, x, y)                            
                    # put text
                    putText(image, dir_total, ear, x, y)
                    
                    # time
                    self.timeCalculation(begin)

                # cv2.imshow('MediaPipe', cv2.flip(image, 0))         # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe', image)            
                vid_writer.write(image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.ok_flag = 0

        cap.release()
        cv2.destroyAllWindows()
        print(self.num)      

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=ROOT / 'shorts12.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--save-path', type=str, default=ROOT / 'run.mp4', help='file/dir/URL/glob')
    args = parser.parse_args()
    return args

def main(args):
    A = GazeEstimation(args)
    A.run()

if __name__ == "__main__":
    args = parse_args()
    main(args)











