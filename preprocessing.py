from pydarknet import Detector,Image
import cv2
import numpy as np
import glob

cfgpath = "yolo/yolo-obj.cfg"
weightpath = "yolo/yolo-obj_1500.weights"
objpath = "yolo/obj.data"

"""
input, imagelist: a list of paths where images can be found
output, fins and flukes:  lists/rather tables (sorted by fin and fluke labels) with line format imagepath and bounding box in the format of x1,y1,x2,y2 
"""
def preprocessImages(imagelist):
  #create YOLOv3 Network
  net = Detector(bytes(cfgpath, encoding="utf-8"), bytes(weightpath, encoding="utf-8"), 0, bytes(objpath,encoding="utf-8"))
  fins = []
  flukes = []
  #process imagewise
  for i in imagelist:
    img = cv2.imread(i)
    img_darknet = Image(img)
    results = net.detect(img_darknet)
    #go through each result for the image
    for cat, score, bounds in results:
      label = str(cat).split("'")[1] #clean the label
      x,y,w,h= bounds
      coor = [int(x-w/2),int(y-h/2),int(x+w/2),int(y+h/2)]
      #define the cropped area
      y1 = max(int(y-h/2),0)
      y2 = min(int(y+h/2),img.shape[0])
      x1 = max(int(x-w/2),0)
      x2 = min(int(x+w/2),img.shape[1])
      if label == "fin":
        fins.append([i,[x1,y1,x2,y2]])
      elif label == "fluke":
        flukes.append([i,[x1,y1,x2,y2]])
  return(fins,flukes)

