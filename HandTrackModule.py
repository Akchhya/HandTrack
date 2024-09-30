import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands = 2, detectionCon= 0.5, trackCon = 0.5  ):
        self.mode =  mode
        self.maxHands = maxHands
        self.detectionCon = float(detectionCon)
        self.trackCon = float(trackCon)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
                #for id, lm in enumerate(handLms.landmark):
                    #print(id,lm)
                    #h, w, c = img.shape
                    #cx, cy = int(lm.x*w), int(lm.y*h)
                    #print(id, cx, cy)
                    #if id == 0:
                    #cv2.circle(img, (cx,cy), 15, (255, 0, 255), cv2.FILLED)*/



def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)

        # Change the waitKey value to 1 to allow for continuous display
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()