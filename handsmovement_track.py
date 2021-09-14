import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, pic = cap.read()
    imgRGB = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h,w,c = pic.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 1:
                    cv2.circle(pic, (cx, cy),10,(300, 1, 300), cv2.FILLED)


            mpDraw.draw_landmarks(pic, handLms, mpHands.HAND_CONNECTIONS)

    cTime  = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(pic,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN, 5, (50,205,0),5)


    cv2.imshow("Result", pic)
    cv2.waitKey(1)
