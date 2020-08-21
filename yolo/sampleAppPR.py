from pydarknet import Detector,Image
import cv2
import glob
import numpy as np
import csv

#imagelist = glob.glob("../data/train/*.jpg")
#imagelist.extend(glob.glob("../data/train/*.JPG"))

def absToRelative(x,y,w,h,o_x,o_y):
    n_x = float(x)/float(o_x)
    n_y = float(y)/float(o_y)
    n_w = float(w)/float(o_x)
    n_h = float(h)/float(o_y)
    return(n_x,n_y,n_w,n_h)

def relativeToAbsolute(x,y,w,h,o_x,o_y):
    n_x = float(x)*float(o_x)
    n_y = float(y)*float(o_y)
    n_w = float(w)*float(o_x)
    n_h = float(h)*float(o_y)
    return(n_x,n_y,n_w,n_h)

def classesMatch(objpath,a,b):
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

def evaluateSet(setpath,cfgpath,weightpath,objpath,objnamespath):
  print(weightpath)
  imagelist = readSet(setpath)
  #print(imagelist)
  net = Detector(bytes(cfgpath, encoding="utf-8"), bytes(weightpath, encoding="utf-8"), 0, bytes(objpath,encoding="utf-8"))
  set_ious = []
  for i in imagelist:
    #print(i)
    img = cv2.imread(i)
    img_darknet = Image(img)
    label = readLabels(str(i[:-3]+ "txt"))
    results = net.detect(img_darknet)
    #print(results)
    coords = []
    cats = []
    for cat, score, bounds in results:
      x,y,w,h= bounds
      #coor = absToRelative(x,y,w,h,img.shape[1],img.shape[0])
      coor = [int(x-w/2),int(y-h/2),int(x+w/2),int(y+h/2)]
      coords.append(coor)
      cats.append(cat)

    if len(label) == 0 and len(coords) == 0: #if no boxes on either
      iou = None #nothing happens
    elif len(label) == 0 and len(coords) > 0:
      set_ious.append(0)
    elif len(coords) == 0: #if no box was detected
      set_ious.append(0)
    elif len(label) > 0 and len(coords) > 0: #if at least one box on either
      img_ious = []
      #get all combination ious and take best matches
      for l in label:
          best_iou = 0
          best_idx = 0
          for i in range(len(coords)):
            if classesMatch(objnamespath,l[0],cats[i]):
              x,y,w,h = relativeToAbsolute(l[1],l[2],l[3],l[4],img.shape[1],img.shape[0])
              groundtruth = [int(x-w/2),int(y-h/2),int(x+w/2),int(y+h/2)]
              iou = getIoU(groundtruth,coords[i])
              if iou > best_iou:
                best_iou = iou
                best_idx = i
          if len(coords) > best_idx:
            del coords[best_idx]
          img_ious.append(best_iou)
      set_ious.extend(img_ious)
    else:
      print("WHAAAT") #this shouldn't happen
      print(label)
      print(coords)
      break
  print(np.mean(set_ious))
  return(set_ious)


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
  i1 = (intersection[3]-intersection[1])
  i2 = (intersection[2]-intersection[0])
  i = i1*i2 
  return i

def getIoU(a,b): #format of a and b is x1,y1,x2,y2
    # a = np.array(a, np.float32)
    b = np.array(b, np.float32)
    intersection = getIntersection(a,b)
    asize = (a[2]-a[0])*(a[3]-a[1])
    bsize = (b[2]-b[0])*(b[3]-b[1])
    if intersection > 0:#
        union = asize + bsize - intersection
    else:
        union = asize + bsize
    return(intersection/union)


def getParts(truth,pred): #truth and pred are both SINGLE BOXES
  tcoor = yoloCoordinatesToCorners(truth[1],truth[2],truth[3],truth[4]) #0th element is class label
  pcoor = yoloCoordinatesToCorners(pred[0],pred[1],pred[2],pred[3])
  iou = getIoU(tcoor,pcoor)
  return iou

def getPrecisionRecall(truth,pred):
  tp = 0
  fp = 0
  fn = 0
  if len(truth) == 1:
    iou = getParts(truth[0],pred[0])
    #print(iou)
    if iou > 0.5:
      tp += 1
    else:
      fp += 1
  else:
    for i in range(len(truth)):
      if i < len(pred):
        iou = getParts(truth[i],pred[i])
        #print(iou)
        if iou > 0.5:
          tp += 1
        else:
          fp += 1
      else: 
        fn += 1
  if not tp == 0:
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
  else:
    #print("MIST")
    precision = 0
    recall = 0
  return(iou,precision, recall)


def readLabels(filename):
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

def yoloCoordinatesToCorners(x,y,w,h): #takes in yolo coordinates x,y of center, w,h and returns the min and max x and y values
  x = float(x)
  y = float(y)
  w = float(w)
  h = float(h) 
  up = float(y-h/2) #upper
  lo = float(y+h/2) #lower
  ri = float(x+w/2) #right
  le = float(x-w/2) #left
  return(le,ri,up,lo) #LEFT RIGHT UPPER LOWER

def bla():

  #net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0, bytes("cfg/coco.data",encoding="utf-8"))
  net = Detector(bytes("yolo-obj.cfg", encoding="utf-8"), bytes("yolo-obj_last.weights", encoding="utf-8"), 0, bytes("obj.data",encoding="utf-8"))
  counter = 0
  for i in imagelist:
      img = cv2.imread(i)
      img_darknet = Image(img)
      results = net.detect(img_darknet)
      #groundtruth = readLabels("labels/" + str(i[7:-3]) + "txt")
      #coords = []
      #crops = []
      for cat, score, bounds in results:
          x,y,w,h= bounds
          y1 = max(int(y-h/2),0)
          y2 = min(int(y+h/2),img.shape[0])
          x1 = max(int(x-w/2),0)
          x2 = min(int(x+w/2),img.shape[1])
          crop = img[y1:y2,x1:x2]
          #print(crop.shape)
          #crops.append(crop)
          #coor = absToRelative(x,y,w,h,img.shape[1],img.shape[0])
          #coords.append(coor)
          #cv2.rectangle(img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (255,0,0), thickness=2)
          #cv2.putText(img,str(cat.decode("utf-8")),(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0))
          cv2.imwrite(str(i[:14] + "crops/" + i[14:]),crop)
      counter += 1
      if counter%10 == 0:
      	print(counter)
      """
      if not len(groundtruth) == 0:
          #print("hallo")
          iou, p, r = getIoU(groundtruth,coords)
      else:
          iou = 0
          p = 0
          r = 0
      cv2.imwrite(str(i[:-4] + "iou" + str(iou) + "p" + str(p) + "r" + str(r) + "_output_.jpg"), img)
      """
      #method for saving all crops needed

def main():
  
    for i in range(600,4000,100):
      result = evaluateSet("../data/yolo/val.txt","yolo-obj.cfg",str("/tmp/yoloweights/yolo-obj_" + str(i) + ".weights"),"obj.data","data/obj.names")
      file = []
      with open('val_results.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
          file.append(row)
      with open('val_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(file)
        writer.writerow([i, np.mean(result), result])


if __name__ == "__main__":
    main()