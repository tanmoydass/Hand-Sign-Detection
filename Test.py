import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

class SignLanguageDetection_Test:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

    imgSize = 300
    offset = 20
    labels = ["Call","Dislike","Fine","Good Job","Good Luck","Power","Stop","Victory"]
    text = "Hand is not detecting"

    while True:
        success, img = cap.read()
        imgOutput = img.copy()
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
                    prediction, index = classifier.getPrediction(imgBack, draw=False)
                    cv2.putText(imgOutput, labels[index], (x -offset, y - 40), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 7)
                except Exception as e:
                    print(str(e))
                    print(text)
                    cv2.putText(imgOutput, text, (x + 40, y +100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                # print(prediction,index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                try:
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgBack[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgBack, draw=False)
                    cv2.putText(imgOutput, labels[index], (x -offset, y - 40), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 7)
                except Exception as e:
                    print(str(e))
                    print(text)
                    cv2.putText(imgOutput, text, (x + 40, y + 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                # print(prediction, index)

        cv2.imshow("Sign Language Recognition", imgOutput)
        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()
