import cv2
import numpy as np
from ex1.ex1 import binarization

threshold = 1000

def mouseHandle(x, y, flags, params):
    pass


if __name__ == "__main__":
    I = cv2.imread("images/objects.jpg", cv2.IMREAD_GRAYSCALE)
    J = binarization(I)
    '''
    contours, detected, b = countComponents(J, 0)
    selected_contours = []

    for i in range(1, detected):
        print i
        contour = np.array(contours[i])
        try:
            if cv2.contourArea(contour) > threshold:
                selected_contours.append(contour)
        except:
            print "failed analysing contour %d" % i

    '''
    detected, h = cv2.findContours(J, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    selected_contours = []

    for i in range(1, len(detected)):

        contour = np.array(detected[i])
        try:
            if cv2.contourArea(contour) > threshold:
                selected_contours.append(contour)
        except:
            print "failed analysing contour %d" % i

    new_img = np.zeros(I.shape, dtype=I.dtype)
    new_img.fill(255)

    cv2.drawContours(new_img, selected_contours, -1, 2, 8)

    cv2.imshow('Contours', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
