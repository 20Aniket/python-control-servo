import cv2
import mediapipe as mp
import time
import math
from pyfirmata import Arduino, util

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0, trackCon=0.5, arduino_port='/dev/cu.usbmodem101'):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.middleFingerId = 12
        self.lastMiddleFingerX = 0
        self.direction = ""
        self.tipIds = [4, 8, 12, 16, 20]
        
        self.board = Arduino(arduino_port)  # Connect to Arduino board
        self.servo1_pin = 12
        self.servo2_pin = 13
        self.servo1 = self.board.get_pin(f"d:{self.servo1_pin}:s")
        self.servo2 = self.board.get_pin(f"d:{self.servo2_pin}:s")

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findDirection(self, lmList, prevMiddleFingerX, prevMiddleFingerY):
        middleFingerX = lmList[self.middleFingerId][1]
        middleFingerY = lmList[self.middleFingerId][2]

        horizontal_threshold = 20
        vertical_threshold = 20

        directionX = ""
        directionY = ""

        if abs(middleFingerX - prevMiddleFingerX) > horizontal_threshold:
            directionX = "Left" if middleFingerX < prevMiddleFingerX else "Right"

        if abs(middleFingerY - prevMiddleFingerY) > vertical_threshold:
            directionY = "Up" if middleFingerY < prevMiddleFingerY else "Down"

        return directionX, directionY, middleFingerX, middleFingerY

    def findPosition(self, img, draw=True):
        bboxes = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            for handNo, handLms in enumerate(self.results.multi_hand_landmarks):
                xList = []
                yList = []
                bbox = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    self.lmList.append([id, cx, cy, handNo])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax
                bboxes.append(bbox)
                if draw:
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
        return self.lmList, bboxes

    def fingersUp(self):
        fingers = []
        if self.lmList[self.middleFingerId][1] > self.lastMiddleFingerX:
            fingers.append(1)
        else:
            fingers.append(0)
        self.lastMiddleFingerX = self.lmList[self.middleFingerId][1]
        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
    cap = cv2.VideoCapture(1)
    detector = HandDetector()
    frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    detector.servo1.write(90)
    detector.servo2.write(90)

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        img = detector.findHands(img, draw=True)
        lmList, bboxes = detector.findPosition(img, draw=True)

        if lmList:
            middleFingerX = lmList[detector.middleFingerId][1]
            middleFingerY = lmList[detector.middleFingerId][2]

            # Map the hand position to the servo angle
            servo1_angle = mapValue(middleFingerX, 0, frameWidth, 0, 180)
            servo2_angle = mapValue(middleFingerY, 0, frameHeight, 0, 180)

            # Ensure the servo angles are within the valid range
            servo1_angle = max(0, min(180, servo1_angle))
            servo2_angle = max(0, min(180, servo2_angle))

            # Write the angles to the servos
            detector.servo1.write(servo1_angle)
            detector.servo2.write(servo2_angle)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        for bbox in bboxes:
            cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            detector.servo1.write(90)  # Set servo angle to the middle position before exiting
            detector.servo2.write(90)  # Set servo angle to the middle position before exiting
            break

    cap.release()
    cv2.destroyAllWindows()

def mapValue(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

if __name__ == "__main__":
    main()