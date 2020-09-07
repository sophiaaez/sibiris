import cv2
import glob
import numpy as np
import csv
import matplotlib.pyplot as plt
from pydarknet import Detector,Image

def classesMatch(a,b,objpath="data/obj.names"):
  try:
    f = open(objpath, "r")
    lines = f.read().split('\n')
    for l in range(len(lines)):
      lines[l] = lines[l].split()
    f.close()
  except FileNotFoundError:
    lines = []
  if a == b:
    return True
  elif str(a) in str(b):
    return True
  elif str(b) in str(a):
    return True
  elif (len(a) == 1 and str(lines[int(a)][0]) in str(b)):
    return True
  elif(len(b) == 1 and str(a) in str(lines[int(b)][0])):
    return True
  else:
    return False
    
def evaluateResultsMAP(setpath,resultspath):
    imagelist = readSet(setpath)
    results = np.array(readResults(resultspath))
    iouthresholds = np.divide(list(range(50,100,5)),100.)
    pn = {}
    for i in iouthresholds:
      pn[i] = []
    order = np.argsort(results[:,2],axis=0)[::-1][1:] #sort by descending score/confidence value
    for o in order: 
        i = results[o,0]
        img = cv2.imread(i)
        label = readLabel(str(i[:-3]+ "txt"))
        pred = np.array((results[o,3])[1:-1].split(','),np.float32)
        if len(label) == 0: #if no ground truth but detected box, coz otherwise we wouldnt be here in the first place
            for threshold in iouthresholds:
              pn[threshold].append("FP")
        else: #if a box was detected
            best_iou = 0
            for l in label: #find bestmatching label for detected box
                if classesMatch(l[0],results[o,1]):
                    x,y,w,h = relativeToAbsolute(l[1],l[2],l[3],l[4],img.shape[1],img.shape[0]) #label struktur original x,y,w,h in relativ
                    groundtruth = [float(x-w/2),float(y-h/2),float(x+w/2),float(y+h/2)]
                    coords = [float(pred[0]-pred[2]/2),float(pred[1]-pred[3]/2),float(pred[0]+pred[2]/2),float(pred[1]+pred[3]/2)]
                    iou = getIoU(groundtruth,coords)
                    if iou > best_iou:
                        best_iou = iou
            for threshold in iouthresholds:
              if iou >= threshold:
                  pn[threshold].append("TP")
              else:
                  pn[threshold].append("FP")
    positives = 0
    for i in imagelist:
        label = readLabel(str(i[:-3]+ "txt"))
        positives += len(label)
    aps = {}
    sap = 0 #sum of average precisions
    for threshold in iouthresholds:
      precs,recs = getPrecisionRecall(pn[threshold],positives)
      smooth,ap = getAP(precs,recs)
      aps[threshold] = ap 
      sap += ap
    m_ap = sap/(len(iouthresholds))
    return (m_ap,aps[0.5], aps[0.75]) #map, ap50, ap75


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

def getIntersection(a,b):
  intersection = [0,0,0,0]
  if a[0] > b[0]: #left
    intersection[0] = a[0]
  else:
    intersection[0] = b[0]
  if a[1] > b[1]: #down
    intersection[1] = a[1]
  else:
    intersection[1] = b[1]
  if a[2] < b[2]: #right
    intersection[2] = a[2]
  else:
    intersection[2] = b[2]
  if a[3] < b[3]: #up
    intersection[3] = a[3]
  else:
    intersection[3] = b[3] 
  i1 = abs(intersection[3]-intersection[1])
  i2 = abs(intersection[2]-intersection[0])
  i = i1*i2 
  if i < 0:
    i = 0
  return i

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

def readLabel(filename):
  try:
    f = open(filename, "r")

    lines = f.read().split('\n')
    for l in range(len(lines)):
      lines[l] = lines[l].split()
      if len(lines[l]) == 0:
        del lines[l]
    f.close()
  except FileNotFoundError:
    lines = []
  return(lines)

def readResults(filename):
	file = []
	with open(filename) as csvfile:
	    reader = csv.reader(csvfile, delimiter=',')
	    for row in reader:
	    	file.append(row)
	return file

def readSet(filename): #reads in files that belong to set, and changes the paths according to the filename, e.g. filename = ../val.txt, original_path= ../obj/bla.jpg -> ../bla.jpg
  try:
    f = open(filename, "r")
    lines = f.read().split('\n')
    separator = '/'
    for l in range(len(lines)):
      if len(lines[l]) == 0:
        del lines[l]
      else:
        path = filename.split('/')[:-1]
        name = lines[l].split('/')[-1]
        lines[l] = str(separator.join(path) + '/' + name)
      
    f.close()
  except FileNotFoundError:
    lines = []
  return(lines)

def relativeToAbsolute(x,y,w,h,o_x,o_y):
    n_x = float(x)*float(o_x)
    n_y = float(y)*float(o_y)
    n_w = float(w)*float(o_x)
    n_h = float(h)*float(o_y)
    return(n_x,n_y,n_w,n_h)

def testEval():
  path = "test_results.csv"
  with open(path, "w") as myfile:
    wr = csv.writer(myfile)
    wr.writerow(["EPOCH","MAP","AP50","AP75"])
    m_ap,ap50,ap75 = evaluateResultsMAP("../data/yolo/test.txt",str("results_test.csv"))
    wr.writerow([m_ap,ap50,ap75])

def valEval():
  path = "eval_results.csv"
  maps = []
  ap50s = []
  ap75s = []
  with open(path, "w") as myfile:
    wr = csv.writer(myfile)
    wr.writerow(["EPOCH","MAP","AP50","AP75"])
    for i in range(600,4000,100):
      m_ap,ap50,ap75 = evaluateResultsMAP("../data/yolo/val.txt",str("results_" + str(i) + ".csv"))
      wr.writerow([i,m_ap,ap50,ap75])
  with open(path, "r") as myfile:
    rd = csv.reader(myfile)
    for row in myfile:
      r = row.split(',')
      if not (r[0] == "EPOCH"):
        maps.append(float(r[1]))
        ap50s.append(float(r[2]))
        ap75s.append(float(r[3]))
  plt.plot(list(range(600,4000,100)),maps,label="mAP")
  plt.plot(list(range(600,4000,100)),ap50s,'r', label="AP50")
  plt.plot(list(range(600,4000,100)),ap75s, 'y', label="AP75")
  plt.legend()
  plt.savefig('eval_results.png')


def main():
  testEval()

if __name__ == "__main__":
    main()