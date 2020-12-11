#!/usr/bin/env python
# coding: utf-8

# ## Load necessary modules
# import keras
import keras
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import glob
import csv

import sys
sys.path.insert(0, '../')


# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

from eval import getAP, getIoU, getPrecisionRecall, getIntersection, readResults

labels_to_names = {0: 'fin', 1:'fluke'}

def classesMatch(a,b):
    if a == b:
        return True
    elif a in labels_to_names and labels_to_names[a] == b:
        return True
    elif b in labels_to_names and a == labels_to_names[b]:
        return True
    elif a in labels_to_names and b in labels_to_names and labels_to_names[a] == labels_to_names[b]:
        return True


"""
Evaluates a Set given a setpath 
Returns a list of results
"""
def evaluateSet(setpath,weightnumber=13):
    gpu = 0
    setup_gpu(gpu)
    model_path = os.path.join('..', 'keras_retinanet', 'resnet50_csv_' + str(weightnumber) + '_final.h5')
    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')
    iset = readSet(setpath) #read set
    imagelist = iset[:,0] # load imagelist
    resultlist = [["image","bounds","score", "label"]]
    for i in imagelist:
        image = read_image_bgr(str("../" + i)) #originally bgr(i)
        if draw:
            image_copy = image.copy()
        image = preprocess_image(image)
        image, scale = resize_image(image)
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        coords = []
        cats = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
          if score < 0.4:
            break
          else:
            resultlist.append([i,box,label,score])
    return resultlist

"""
Calculates the MAP, AP50 and AP75 of a set given the ground truth setpath and the predicted resultpath
Returns the MAP overall and the AP50 and AP75
"""
def evaluateResults(setpath,resultpath):
    iset = readSet(setpath)
    results = readResults(resultpath)
    res = []
    for i in range(len(results)):
        if i > 0:
            r = results[i]
            path = r[0]
            box = r[1].split(' ')
            true_box = []
            for b in box:
                if '.' in b: #numbers all have decimal points
                    if '[' in b:
                        true_box.append(float(b[1:]))
                    elif ']' in b:
                        true_box.append(float(b[:-1]))
                    else:
                        true_box.append(float(b))
            label = int(r[2])
            score = float(r[3])
            res.append([path,true_box,label,score])
    results = np.array(res)
    iouthresholds = np.divide(list(range(50,100,5)),100.)
    pn = {}
    for i in iouthresholds:
        pn[i] = []
    order = np.argsort(results[:,-1],axis=0)[::-1][1:] #sort by descending score/confidence value
    for o in order: 
        i = results[o,0]
        img = cv2.imread(str("../" + i))
        label = findLabelAmongLabels(i,iset)
        pred = np.array(results[o,1])
        if len(label) == 0: #if no ground truth but detected box, coz otherwise we wouldnt be here in the first place
            for threshold in iouthresholds:
              pn[threshold].append("FP")
        else: #if a box was detected
            best_iou = 0
            for l in label: #find bestmatching label for detected box
                if classesMatch(l[-1],results[o,2]):
                    iou = getIoU([l[1],l[2],l[3],l[4]],[results[o,1][0],results[o,1][1],results[o,1][2],results[o,1][3]])
                    if iou > best_iou:
                        best_iou = iou

            for threshold in iouthresholds:
              if best_iou >= threshold:
                  pn[threshold].append("TP")
              else:
                  pn[threshold].append("FP")
    positives = len(iset)
    aps = {}
    sap = 0 #sum of average precisions
    for threshold in iouthresholds:
      precs,recs = getPrecisionRecall(pn[threshold],positives)
      smooth,ap = getAP(precs,recs)
      aps[threshold] = ap 
      sap += ap
    m_ap = sap/(len(iouthresholds))
    return (m_ap,aps[0.5], aps[0.75]) #map, ap50, ap75        

def findLabelAmongLabels(label,labels):
    found = []
    for l in labels:
        if label in l:
            found.append(l)
    return(found)

"""
reads in files that belong to set, and changes the paths according to the filename, e.g. filename = ../val.txt, original_path= ../obj/bla.jpg -> ../bla.jpg
"""
def readSet(filename): #reads in files that belong to set, and changes the paths according to the filename, e.g. filename = ../val.txt, original_path= ../obj/bla.jpg -> ../bla.jpg
    file = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            file.append(row)
    return(np.array(file))

"""
Runs one Image through the network and saves it with the bounding box and label illustrating the output
"""
def saveOutputForImage(image,filepath)
  gpu = 0
  setup_gpu(gpu)
  model_path = os.path.join('..', 'keras_retinanet', 'resnet50_csv_' + str(weightnumber) + '_final.h5')
  # load retinanet model
  model = models.load_model(model_path, backbone_name='resnet50')
  image = read_image_bgr(image)
  image = preprocess_image(image)
  image, scale = resize_image(image)
  boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
  boxes /= scale
  coords = []
  cats = []
  for box, score, label in zip(boxes[0], scores[0], labels[0]):
    if score < 0.4:
      break
    else:
      resultlist.append([i,box,label,score])
      cv2.rectangle(image, (box[0],box[1],box[2],box[3]), (255,0,0), thickness=2)
      cv2.putText(image,str(cat.decode("utf-8")),(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))
  output_name = str(filepath + "output_" + image.split('/')[-1])
  cv2.imwrite(output_name,image)

"""
Evaluates the test set and saves it accordingly
"""
def testEval():
  results = evaluateSet("../data/retinanet/labels_test.csv","13")
  writeResults(results,"retinaresults/TEST_retisults","13")
  with open("retinanet_test_results.csv", "w") as myfile:
    wr = csv.writer(myfile)
    wr.writerow(["MAP","AP50","AP75"])
    m_ap,ap50,ap75 = evaluateResults("../labels_test.csv",str("./retinaresults/TEST_retisults_13.csv"))
    wr.writerow([m_ap,ap50,ap75])

"""
Evaluates all the validation sets and creates a nice, shiny graph
"""
def valEval():
  for j in range(1,36):
      if len(str(j)) == 1:
        i = str("0"+ str(j))
      else:
        i = str(j)
      results = evaluateSet("../data/retinanet/labels_val.csv",i)
      writeResults(results,"./retinaresults/retisults",i)
      #eresults = evaluateResults("../labels_val.csv",str("./results/retisults_" + str(i) + ".csv"))
      #print(eresults)
  with open("reinanet_val_results.csv", "w") as myfile:
    wr = csv.writer(myfile)
    wr.writerow(["EPOCH","MAP","AP50","AP75"])
    for j in range(1,36):
      if len(str(j)) == 1:
        i = str("0"+ str(j))
      else:
        i = str(j)
      m_ap,ap50,ap75 = evaluateResults("../data/retinanet/labels_val.csv",str("./retinaresults/retisults_" + str(i) + ".csv"))
      wr.writerow([i,m_ap,ap50,ap75])

def writeResults(results,path,weightnr):
  path = str(path + "_" + str(weightnr) + ".csv")
  with open(path, "w") as myfile:
    wr = csv.writer(myfile)
    for r in results:
      wr.writerow(r)
