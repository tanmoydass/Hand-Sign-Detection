import cv2
import numpy as np
import time
import math
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = HandDetector(maxHands=1)

imgSize = 300
offset = 20
folder = "Images/"
counter = 0


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgBack = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        ratio = h / w

        if ratio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            try:
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgBack[:, wGap:wCal + wGap] = imgResize
            except Exception as e:
                print(str(e))
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            try:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgBack[hGap:hCal + hGap, :] = imgResize
            except Exception as e:
                print(str(e))

        cv2.imshow("Background", imgBack)
        if x > 0 + offset and y > 0 + offset and w > 0 + offset and h > 0 + offset:
            cv2.imshow("Img Crop", imgCrop)

    cv2.imshow("Total Image", img)

    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgBack)
        print(counter)
    if key == 13:
        break

cap.release()
cv2.destroyAllWindows()
