import cv2
import glob
import numpy as np
import csv
import matplotlib.pyplot as plt
from pydarknet import Detector,Image

"""
Calculates the average precision based on the precision and recall values,
which are essentially the output of getPrecisionRecall
Returns the 101pt interpolation curve and a single average precision value
"""
def getAP(prec,rec):
  #smooth
  prec0 = prec.copy()
  prec0.append(0.0)
  smoothprec = np.zeros(101) #smoothed and ready for easy 101pt interpolation
  for idx in range(101):
    i = (100-idx)/100.
    val = 0
    for re_idx in range(len(rec)): #go through recs
      re_i = len(rec)-re_idx-1 #from back to front
      if rec[re_i] >= i: # if value there is larger than i
        val = max(prec0[re_i:])
        #break
    smoothprec[100-idx] = val
  #quick 101 pt interpolation
  ap = np.mean(smoothprec)
  return(smoothprec,ap)

"""
Calculates the intersection of two boxes a and b,
both arrays are in the format x1,y1,x2,y2, where x1,y1 and x2,y2 are 
the upmost left and downmost right corner
Returns a single value for the Intersection amount in pixels
"""
def getIntersection(a,b): #each in format x1,y1,x2,y2
  intersection = [0,0,0,0]
  #left -> 
  if b[0] <= a[0] and a[0] <= b[2]:
    intersection[0] = a[0]
  elif a[0] <= b[0] and b[0] <= a[2]:
    intersection[0] = b[0]
  else: 
    return 0
  #down ->
  if b[1] <= a[1] and a[1] <= b[3]:
    intersection[1] = a[1]
  elif a[1] <= b[1] and b[1] <= a[3]:
    intersection[1] = b[1]
  else:
    return 0
  #right ->
  if b[0] <= a[2] and a[2] <= b[2]: 
    intersection[2] = a[2]
  elif a[0] <= b[2] and b[2] <= a[2]:
    intersection[2] = b[2]
  else:
    return 0
  #up ->
  if b[0] <= a[3] and a[3] <= b[3]: #up
    intersection[3] = a[3]
  elif a[0] <= b[3] and b[3] <= a[3]:
    intersection[3] = b[3] 
  else:
    return 0
  i1 = intersection[3]-intersection[1]
  i2 = intersection[2]-intersection[0]
  i = i1*i2 
  return i

"""
Calculates the IoU Intersection over Union for the two boxes a and b,
both arrays are in the format x1,y1,x2,y2, where x1,y1 and x2,y2 are 
the upmost left and downmost right corner
Returns a single IoU value
"""
def getIoU(a,b): #format of a and b is x1,y1,x2,y2
    a = np.array(a, np.float32)
    b = np.array(b, np.float32)
    intersection = getIntersection(a,b)
    asize = (a[2]-a[0])*(a[3]-a[1])
    bsize = (b[2]-b[0])*(b[3]-b[1])
    if intersection > 0:#
        union = asize + bsize - intersection
    else:
        union = asize + bsize
    return(intersection/union)

"""
Calculates the precision and recall values/curve given plist that contains only "TP" and "FP" items
this list was created by predictions that are ordered based on score
and positives, the number of all positives based on the ground truth
Returns tuple of lists for precisions and recalls
"""
def getPrecisionRecall(plist,positives):
    tp = 0
    fp = 0
    precs = []
    recs = []
    for e in plist:
        if e == "TP":
            tp += 1
        elif e == "FP":
            fp += 1
        precision = tp/(tp+fp)
        precs.append(precision)
        recall = tp/(positives)
        recs.append(recall)
    return(precs,recs)

def readResults(filename):
	file = []
	with open(filename) as csvfile:
	    reader = csv.reader(csvfile, delimiter=',')
	    for row in reader:
	    	file.append(row)
	return file

"""
converts relative to absolute coordinates,
x = point of box (relative), y = point of box (relative)
w = width of box (relative), h = height of box (relative)
o_x = original width of image, o_y = original height of image
"""
def relativeToAbsolute(x,y,w,h,o_x,o_y):
    n_x = float(x)*float(o_x)
    n_y = float(y)*float(o_y)
    n_w = float(w)*float(o_x)
    n_h = float(h)*float(o_y)
    return(n_x,n_y,n_w,n_h)

