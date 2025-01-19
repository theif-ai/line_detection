import cv2
import numpy as np

width, height = 1000, 1000
mask_left_list = []
mask_right_list = []
left_line = []
right_line = []
loop_num = 20
limit_x_max = 48
limit_y_min = 30
sensitivity = 50000

SCALE = 1.0 # 글자 크기
COLOR = (0,255,0)
THICKNESS = 2

for i in range(loop_num):
    mask_left_list.append(1)
    mask_right_list.append(1)
    left_line.append(1)
    right_line.append(1)


def perspective_transform(image):
    
    
    src = np.array([[534,464],[750,466],[1195,635],[45,660]],dtype = np.float32)
    dst = np.array([[0,0],[width,0],[width,height],[0,height]],dtype = np.float32)
    
    matrix = cv2.getPerspectiveTransform(src,dst)
    lane_image = cv2.warpPerspective(image, matrix,(width,height))
    
    return lane_image

def pre_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #threshold = cv2.adaptiveThreshold(blur, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 101, 10)
    ret, otsu = cv2.threshold(blur,-1,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #ret, threshold = cv2.threshold(blur, 190, 255, cv2.THRESH_BINARY)
    #canny = cv2.Canny(blur, 50, 50)
    
    return otsu

def Contours(image):
    contours, hieracy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    temp = np.zeros((height, width), dtype = np.uint8)
    COLOR = (255,255,255)
    # cv2.drawContours(temp, contours, -1, COLOR, 2)
    for cnt in contours:
        
        x, y , width_rect, height_rect = cv2.boundingRect(cnt)
        
        if width_rect > limit_x_max or height_rect < limit_y_min:
            continue
        cv2.rectangle(temp,(x,y),(x+width_rect,y+height_rect),COLOR,2)
        
        return temp, (x+width_rect)/2
    
    return temp, -1

def mask(image):
    slice_num = int(1000/loop_num)
    
    for i in range(loop_num):
        mask_left_list[i] = np.zeros_like(image)
        polygons = np.array([
        [(0,slice_num*i),(500,slice_num*i),(500,slice_num*(i+1)),(0,slice_num*(i+1))]
        ])
        cv2.fillPoly(mask_left_list[i], polygons, 255)
        mask_left_list[i] = cv2.bitwise_and(image, mask_left_list[i])
        
    for i in range(loop_num):
        mask_right_list[i] = np.zeros_like(image)
        polygons = np.array([
        [(500,slice_num*i),(1000,slice_num*i),(1000,slice_num*(i+1)),(500,slice_num*(i+1))]
        ])
        cv2.fillPoly(mask_right_list[i], polygons, 255)
        mask_right_list[i] = cv2.bitwise_and(image, mask_right_list[i])
        
    return 0
    
def line_change(right_cnt, left_cnt):
    if right_cnt >10 :
        cv2.putText(frame,'you can change line',(859,500),cv2.FONT_HERSHEY_SIMPLEX,SCALE,(0,255,0),THICKNESS)
    else :
        cv2.putText(frame,'you can`t change line',(859,500),cv2.FONT_HERSHEY_SIMPLEX,SCALE,(0,0,255),THICKNESS)
        
    if left_cnt >10 :
        cv2.putText(frame,'you can change line',(150,500),cv2.FONT_HERSHEY_SIMPLEX,SCALE,(0,255,0),THICKNESS)
    else :
        cv2.putText(frame,'you can`t change line',(150,500),cv2.FONT_HERSHEY_SIMPLEX,SCALE,(0,0,255),THICKNESS)
    return 0

def steer(right_cnt, left_cnt):
    sum_right = 0
    sum_left = 0
    n = 0
    z = 0
    for i in range(loop_num):
        if right_line[i] == -1 :
            continue
        
        if n == 0:
            right = right_line[i]
            n = n + 1
            continue
        
        elif right_line[i] > right:
            sum_right = sum_right + right_line[i]
            
        elif right_line[i] < right:
            sum_right = sum_right - right_line[i]
           
        if left_line[i] == -1:
            continue
        
        if z == 0:
            left = left_line[i]
            z = z + 1
            continue
            
        elif left_line[i] > left:
            sum_left = sum_left + left_line[i]
        
        elif left_line[i] < left:
             sum_left = sum_left - left_line[i]
                
    su = round((sum_left+sum_right)/(40-(right_cnt+left_cnt)))
                
    if sum_left * sum_right <= 0:
        cv2.putText(frame,'straight',(650,500),cv2.FONT_HERSHEY_SIMPLEX,SCALE,(255,0,0),THICKNESS)
        
    elif sum_left + sum_right > sensitivity :
        cv2.putText(frame,str(su),(650,500),cv2.FONT_HERSHEY_SIMPLEX,SCALE,(255,0,0),THICKNESS)
        
    elif sum_left +sum_right < sensitivity :
        cv2.putText(frame,str(su),(650,500),cv2.FONT_HERSHEY_SIMPLEX,SCALE,(255,0,0),THICKNESS)
    
    else: 
        cv2.putText(frame,'straight',(650,500),cv2.FONT_HERSHEY_SIMPLEX,SCALE,(255,0,0),THICKNESS)
        
    return 0
        
    
        

cap = cv2.VideoCapture('project_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_copy = frame.copy()
    
    lane_image = perspective_transform(frame_copy)
    pre_frame = pre_processing(lane_image)
    mask(pre_frame)
    right_cnt = 0
    left_cnt = 0
    for i in range(loop_num):
        mask_left_list[i], left_line[i] = Contours(mask_left_list[i])
        mask_right_list[i] , right_line[i]= Contours(mask_right_list[i])
        
        if right_line[i] == -1:
            right_cnt = right_cnt + 1
                
        if left_line[i] == -1:
            left_cnt = left_cnt + 1
        
        if i == 0:
            combo_left = mask_left_list[i]
            combo_right = mask_right_list[i]
            
        else:
            combo_left = cv2.addWeighted(combo_left, 1, mask_left_list[i], 1, 1)
            combo_right = cv2.addWeighted(combo_right, 1, mask_right_list[i], 1, 1)
            
    line_change(right_cnt, left_cnt)
    #steer(right_cnt, left_cnt)
            
    combo_image = cv2.addWeighted(combo_left, 1, combo_right, 1, 1)
    
    cv2.imshow('video',frame)
    cv2.imshow('lane_image',lane_image)
    cv2.imshow('pre_frame',pre_frame)
    cv2.imshow('combo_image',combo_image)
    if cv2.waitKey(25) == ord('q'):
        break
cv2.destroyAllWindows()