import cv2
import time
import os
from handTrackingModule import *

def check_total_finger(total_finger,frame, fingers_path):
        if total_finger == 0:
            frame[0:height, 0:width] = fingers_path[0]
        elif total_finger == 1:
            frame[0:height, 0:width] = fingers_path[1]
        elif total_finger == 2:
            frame[0:height, 0:width] = fingers_path[2]
        elif total_finger == 3:
            frame[0:height, 0:width] = fingers_path[3]
        elif total_finger == 4:
            frame[0:height, 0:width] = fingers_path[4]
        else:
            frame[0:height, 0:width] = fingers_path[5]



hand_detector = HandDetector()
fingers = os.listdir(os.path.join("fingers"))
fingers_path = []
for fingerPath in fingers:
    image = cv2.resize(cv2.imread(os.path.join("fingers", fingerPath)),(0,0), fx=0.75,fy=0.75)
    fingers_path.append(image)

camWidth = 1250
camHeight = 750
cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

tip_finger_ids = [4,8,12,16,20]

run = True
while run:
    ret, frame = cap.read()

    height, width, channel = fingers_path[0].shape
  
    frame = hand_detector.find_hands(frame)
    position_list = hand_detector.find_position(frame,draw=False)

    if len(position_list)!=0:
        is_finger_open = []

        if position_list[4][1] > position_list[3][1]:
            is_finger_open.append(1)
        else:
            is_finger_open.append(0)

        for id in  range(1,5):
            if position_list[tip_finger_ids[id]][2] < position_list[tip_finger_ids[id]-2][2] and tip_finger_ids[id] != 4:
                is_finger_open.append(1)
            else:
                is_finger_open.append(0)
        total_finger = is_finger_open.count(1)
        #print(is_finger_open)
        print(total_finger)
        check_total_finger(total_finger,frame,fingers_path)
       



    hand_detector.show_fps(frame, camWidth - 100, 30)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        run = False
cap.release()
cv2.destroyAllWindows()



