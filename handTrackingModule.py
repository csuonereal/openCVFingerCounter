import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, static_mode=False, maxHands=2, detectionConfident=0.5, trackConfident=0.5):
        self.static_mode = static_mode
        self.maxHands = maxHands
        self.detectionConfident = detectionConfident
        self.trackConfident = trackConfident
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(self.static_mode, self.maxHands, self.detectionConfident, self.trackConfident)
        self.pTime = 0
        self.cTime = 0

    def show_fps(self, frame, coordX=30,coordY=30):
        self.cTime = time.time()
        self.fps = 1 / (self.cTime - self.pTime)
        self.pTime = self.cTime
        cv2.putText(frame, "FPS:"+str(+int(self.fps)), (coordX, coordY), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255), 1)

    def find_hands(self, frame, draw=True):
        img_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for hand_lmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, hand_lmarks, self.mpHands.HAND_CONNECTIONS)
        return frame
    def find_position(self, frame, hand_number=0, draw=True):
        pos_list = []
        if self.results.multi_hand_landmarks:
            for hand_lmarks in self.results.multi_hand_landmarks:
                for id, lm in enumerate(hand_lmarks.landmark):
                    height, width, channel = frame.shape
                    cx, cy = int(lm.x*width),int(lm.y*height)
                    pos_list.append([id,cx,cy])
                    if draw:
                        cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return pos_list


def main():
    cap = cv2.VideoCapture(0)
    hand_detector = HandDetector()
    while True:
        ret,frame = cap.read()
        frame = hand_detector.find_hands(frame)
        lmList = hand_detector.find_position(frame)
        if len(lmList) != 0:
            print(lmList[4])
        hand_detector.find_hands(frame)
        hand_detector.show_fps(frame)
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
