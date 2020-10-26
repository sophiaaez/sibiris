from pydarknet import Detector,Image
import cv2
import glob
import numpy as np
import csv


def relativeToAbsolute(x,y,w,h,o_x,o_y):
    n_x = float(x)*float(o_x)
    n_y = float(y)*float(o_y)
    n_w = float(w)*float(o_x)
    n_h = float(h)*float(o_y)
    return(n_x,n_y,n_w,n_h)

def fakeCropAllInFolder(folder,cfgpath,weightpath,objpath):
  imagelist = glob.glob(str(folder + "*.jpg"))
  imagelist.extend(glob.glob(str(folder + "*.JPG")))
  imagelist.extend(glob.glob(str(folder + "*.jpeg")))
  imagelist.extend(glob.glob(str(folder + "*.JPEG")))
  delimiter = '/'

  path = "./test_cropped.csv"
  net = Detector(bytes(cfgpath, encoding="utf-8"), bytes(weightpath, encoding="utf-8"), 0, bytes(objpath,encoding="utf-8"))
  counter = 0
  rows = []

  for i in imagelist:
    counter += 1
    img = cv2.imread(i)
    img_darknet = Image(img)
    results = net.detect(img_darknet)
    for cat, score, bounds in results:
      x,y,w,h= bounds
      y1 = max(int(y-h/2),0)
      y2 = min(int(y+h/2),img.shape[0])
      x1 = max(int(x-w/2),0)
      x2 = min(int(x+w/2),img.shape[1])
      rows.append([i,[x1,y1,x2,y2]])
    if counter %1000 == 0:
      print(counter)
      with open(path, "a") as myfile:
        wr = csv.writer(myfile)
        for r in rows:
          wr.writerow(r)
      rows = []

def fakeCropAllInFolderAFTER(folder,cfgpath,weightpath,objpath,after):
  imagelist = glob.glob(str(folder + "*.jpg"))
  imagelist.extend(glob.glob(str(folder + "*.JPG")))
  imagelist.extend(glob.glob(str(folder + "*.jpeg")))
  imagelist.extend(glob.glob(str(folder + "*.JPEG")))
  delimiter = '/'
  index = imagelist.index(after)
  imagelist = imagelist[index+1:]
  print(len(imagelist))

  path = "./train_cropped.csv"
  net = Detector(bytes(cfgpath, encoding="utf-8"), bytes(weightpath, encoding="utf-8"), 0, bytes(objpath,encoding="utf-8"))
  counter = 0
  rows = []

  for i in imagelist:
    counter += 1
    img = cv2.imread(i)
    img_darknet = Image(img)
    results = net.detect(img_darknet)
    for cat, score, bounds in results:
      x,y,w,h= bounds
      y1 = max(int(y-h/2),0)
      y2 = min(int(y+h/2),img.shape[0])
      x1 = max(int(x-w/2),0)
      x2 = min(int(x+w/2),img.shape[1])
      rows.append([i,[x1,y1,x2,y2]])
    if counter %1000 == 0:
      print(counter)
      with open(path, "a") as myfile:
        wr = csv.writer(myfile)
        for r in rows:
          wr.writerow(r)
      rows = []
  with open(path, "a") as myfile:
        wr = csv.writer(myfile)
        for r in rows:
          wr.writerow(r)




def cropAllInFolder(folder,cfgpath,weightpath,objpath,objnamespath):
  imagelist = glob.glob(str(folder + "*.jpg"))
  imagelist.extend(glob.glob(str(folder + "*.JPG")))
  imagelist.extend(glob.glob(str(folder + "*.jpeg")))
  imagelist.extend(glob.glob(str(folder + "*.JPEG")))
  delimiter = '/'

  net = Detector(bytes(cfgpath, encoding="utf-8"), bytes(weightpath, encoding="utf-8"), 0, bytes(objpath,encoding="utf-8"))
  set_ious = []
  for i in imagelist:
    img = cv2.imread(i)
    img_darknet = Image(img)
    #label = readLabels(str(i[:-3]+ "txt"))
    results = net.detect(img_darknet)
    coords = []
    cats = []
    path = i.split(delimiter)
    counter = 0
    for cat, score, bounds in results:
      counter += 1
      x,y,w,h= bounds
      coor = [int(x-w/2),int(y-h/2),int(x+w/2),int(y+h/2)]
      y1 = max(int(y-h/2),0)
      y2 = min(int(y+h/2),img.shape[0])
      x1 = max(int(x-w/2),0)
      x2 = min(int(x+w/2),img.shape[1])
      crop = img[y1:y2,x1:x2]
      if counter == 1:
        cv2.imwrite(str(delimiter.join(path[:-1]) + "/crops/" + path[-1]),crop)
      else:
        cv2.imwrite(str(delimiter.join(path[:-1]) + "/crops/" + path[-1][:-4] + str(counter) + ".jpg"),crop)

def evaluateSet(setpath,cfgpath,weightpath,objpath,objnamespath,draw=False):
  imagelist = readSet(setpath)
  net = Detector(bytes(cfgpath, encoding="utf-8"), bytes(weightpath, encoding="utf-8"), 0, bytes(objpath,encoding="utf-8"))
  resultlist = [["image","cat","score","bounds"]]
  for i in imagelist:
    img = cv2.imread(i)
    img_darknet = Image(img)
    results = net.detect(img_darknet)
    coords = []
    cats = []
    for cat, score, bounds in results:
      x,y,w,h= bounds
      coor = [int(x-w/2),int(y-h/2),int(x+w/2),int(y+h/2)]
      resultlist.append([i,cat,score,bounds])
      coords.append(coor)
      cats.append(cat)
      if draw:
        print("drawing " + str(i))
        cv2.rectangle(img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (255,0,0), thickness=2)
        cv2.putText(img,str(cat.decode("utf-8")),(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))
      """if draw:
      for l in label:
        x,y,w,h = relativeToAbsolute(l[1],l[2],l[3],l[4],img.shape[1],img.shape[0])
        cv2.rectangle(img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0,255,0), thickness=2)
        cv2.putText(img,str(l[0]),(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0))"""
    cv2.imwrite(str(i[:-4] + "_eval.jpg"),img)
  return resultlist

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

def writeResults(results,path,weightnr):
  path = str(path + "_" + str(weightnr) + ".csv")
  print(path)
  with open(path, "w") as myfile:
    wr = csv.writer(myfile)
    for r in results:
      wr.writerow(r)

def test():
  print("TEST COMPLETED")

def main():
  #fakeCropAllInFolderAFTER("../data/test/","yolo-obj.cfg","yolo-obj_1500.weights","obj.data","../data/test/dff8065d4.jpg")
  #fakeCropAllInFolderAFTER("../data/kaggle/","yolo-obj.cfg","yolo-obj_1500.weights","obj.data","../data/kaggle/fc0980b84.jpg")

if __name__ == "__main__":
    main()