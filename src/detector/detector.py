import cv2
import numpy as np
import imutils
import pytesseract
import json
import os

pytesseract.pytesseract.tesseract_cmd = r'../TesseractOCR/tesseract.exe'




class NumberPlateDetector():
    def __init__(self, path_to_image):
        self.path = path_to_image
        self.images = os.listdir(self.path)
        self.images.sort()
        print(self.images)

    def write_json(self,new_data, filename='recognitions/recog.json'):
        with open(filename,'r+') as file:
            file_data = json.load(file)
            file_data["Cars"].append(new_data)
            file.seek(0)
            json.dump(file_data, file, indent = 4)


    def start(self):

        for img in self.images:
            print(img)
            self.img_path=img
            self.img = cv2.imread(self.path + img)
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.filter = cv2.bilateralFilter(self.gray, 10, 17, 175)
            self.edged = cv2.Canny(self.filter, 30, 210)
            self.keypoints = cv2.findContours(self.edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            self.contours = imutils.grab_contours(self.keypoints)
            self.contours = sorted(self.contours, key=cv2.contourArea, reverse=True)[:10]
            self.location = None


            for contour in self.contours:
                approx = cv2.approxPolyDP(contour, 10, True)
                if len(approx) == 4:
                    self.location = approx
                    break

            (_, thresh) =cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            mask = np.zeros(thresh.shape, np.uint8)
            new_image=cv2.Canny(self.filter, 40, 80)
            new_image = cv2.drawContours(mask, [self.location], 0,255, -1)
            new_image = cv2.bitwise_and(self.img, self.img, mask=mask)

            (x,y) = np.where(mask==255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = cv2.Canny(self.filter, 40, 30)
            cropped_image = self.gray[x1:x2+1, y1:y2+1]

            text = pytesseract.image_to_string(cropped_image, config='--psm 6')
            #new_image = cv2.putText(new_image, text.upper(), (40,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))



            recognition = {
                        "plateNumber": text.upper(),
                        "path": self.path + self.img_path
                        }

            self.write_json(recognition)

            print(text.upper())
            cv2.waitKey(0)
            cv2.destroyAllWindows()
