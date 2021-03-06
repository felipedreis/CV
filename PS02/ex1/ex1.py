import cv2
import numpy as np
from Queue import Queue
import sys

T = 127
Gmax = 255
adj4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
adj8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def Adj(u, adj):
    return [(u[0] + v[0], u[1] + v[1]) for v in adj]

def binarization(I):
    ret, img = cv2.threshold(I, 0, Gmax, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img


def visit(J, s, visited, color, detect, adjType=adj4):
    Q = Queue()
    Q.put(s)
    points = []
    while not Q.empty():
        u = Q.get()
        for v in Adj(u, adjType):
            if v[0] < 0 or v[1] < 0 or v[0] >= J.shape[0] or v[1] >= J.shape[1]:
                continue
            if not visited[v] and J[v] == detect:
                points.append((v[1], v[0]))
                visited[v] = color
                Q.put(v)

    return points


def countComponents(J, detect):
    """
    :param J: binary image
    :param detect: color you want to detect with 8-adjacency, the opposite color will be detected with 4-adjacency
    :return: the objects detected with 8-adjacency (contour-vector), the detected objects, and the visitation matrix
    """
    (w, h) = J.shape
    visited = np.zeros((w, h), dtype=np.int32)
    count = 1
    objects = {}

    for i in range(0, w):
        for j in range(0, h):
            if not visited[i, j] and J[i, j] == detect:
                objects[count] = visit(J, (i, j), visited, count, detect)
                count += 1

    detected = count

    for i in range(0, w):
        for j in range(0, h):
            if not visited[i, j] and J[i, j] == detect:
                objects[count] = visit(J, (i, j), visited, count, Gmax - detect, adj4)
                count += 1

    return objects, detected, visited


def mouseHandler(event, x, y, flags, param):
    (objects, visited) = param

    if event == cv2.EVENT_FLAG_LBUTTON:
        objId = visited[y, x]
        if objId == 0:
            return

        contour = np.array(objects[objId])
        convex = cv2.isContourConvex(contour)

        print "Object id: %d" % objId
        print "Area: %.2f" % cv2.contourArea(contour)
        print "Perimeter: %2.f" % cv2.arcLength(contour, convex)

if __name__ == '__main__':

    if len(sys.argv) < 1:
        print "Usage: python ex1.py image_name.png"
        exit(1)

    print "black > white (0)"
    print "white > black (1)"

    objColor = int(raw_input("Choose your option: "))
    if objColor == 1:
        objColor = Gmax

    fileName = sys.argv[1]
    I = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    I = cv2.GaussianBlur(I, (5,5), sigmaX=1)
    J = binarization(I)

    (objects, detected, visited) = countComponents(J, objColor)
    print "Ojbects found = %d" % detected

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', mouseHandler, (objects, visited))
    cv2.imshow('image', J)
    cv2.imwrite('results/binary.png', J)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
