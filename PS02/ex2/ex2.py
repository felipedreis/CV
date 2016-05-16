import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys

from ex1.ex1 import Adj, Gmax, adj4


def cglm(I):
    (w, h) = I.shape
    C = np.zeros((Gmax + 1, Gmax + 1), dtype=np.int32)

    for i in range(0, w):
        for j in range(0, h):
            c1 = I[i, j]

            for u in Adj((i, j), adj4):
                if u[0] < 0 or u[1] < 0 or u[0] >= I.shape[0] or u[1] >= I.shape[1]:
                    continue
                c2 = I[u]
                C[c1, c2] += 1

    return C


def measure(C):
    (w,h) = C.shape
    homogeneity = 0
    uniformity = 0
    for u in range(0, w):
        for v in range(0, h):
            homogeneity += (C[u, v] / float(1 + np.abs(u - v)))
            uniformity += np.power(C[u, v], 2)

    return homogeneity, uniformity


def smooth(I, n=30):
    (w, h) = I.shape

    S = np.copy(I)
    R = np.zeros((w, h), dtype=np.int32)
    Hs = np.zeros((n, 1), dtype=np.float32)
    Us = np.zeros((n, 1), dtype=np.float32)
    Hr = np.zeros((n, 1), dtype=np.float32)
    Ur = np.zeros((n, 1), dtype=np.float32)

    for i in range(0, n):
        print "calculating for i = ", i
        S = cv2.boxFilter(S, ddepth=0, ksize=(3, 3))
        R = I - S
        Cs = cglm(S)
        Cr = cglm(R)

        Hs[i], Us[i] = measure(Cs)
        Hr[i], Ur[i] = measure(Cr)

    return S, R, (Hs, Us), (Hr, Ur)

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print "Usage: python ex2.py image_name.png result.png"
        exit(1)

    fileName = sys.argv[1]
    resfile = sys.argv[2]

    I = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)

    S, R, (Hs, Us), (Hr, Ur) = smooth(I)
    Co = cglm(I)
    T = np.sum(Co)
    Hs /= T
    Us /= T
    Ur /= T
    Us /= T

    x = np.arange(0, 30)
    fig = plt.figure(figsize=(14, 9), dpi=80)
    fig.suptitle(fileName)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(x, Hs, 'r--')
    ax1.set_title("Homogeneity of S")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(x, Us, 'bs')
    ax2.set_title("Uniformity of S")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(x, Hr, 'r--')
    ax3.set_title("Homogeneity of R")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(x, Ur, 'bs')
    ax4.set_title("Uniformity of R")
    plt.savefig(resfile)
    plt.show()
