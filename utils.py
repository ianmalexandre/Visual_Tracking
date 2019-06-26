from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import os
import numpy as np 
import itertools
import math


def intersect(a, b):
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx>=0) and (dy>=0):
        return dx*dy
    return 0

def union(a,b):
    area1 = abs((a[0] - a[2]) * (a[1] - a[3]))
    area2 = abs((b[0] - b[2]) * (b[1] - b[3]))
    #area1 = abs(a[2]*a[3])
    #area2 = abs(b[2]*b[3])
    return area1 + area2

def jaccard(a, b):

    inter = intersect(a, b)
    un = union(a,b) - inter
    if un > 0:
        return inter/un
    return 0


def evaluate(groundTruth, AllJaccards, errors, name, testname):
    new_test = np.delete(AllJaccards, AllJaccards[0])
    
    totalJaccard = np.mean(AllJaccards)
    aux = AllJaccards.size
    print("Proporção de falhas por frames: ", errors/aux)
    print("Final Jaccard Index: ", totalJaccard)
    print("Storing the results on result" + name + "_" + testname + ".txt")
    with open('result' + name + '_' + testname + '.txt', 'a') as ResultFile:
        ResultFile.write(name + "\t Test" + testname + "\n------------------\n")
        ResultFile.write("{}\n{}\n---------\n".format(errors/aux, totalJaccard))