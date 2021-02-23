from pydarknet import Detector,Image
import cv2
import glob
import csv
from eval import getAP, getIoU, getPrecisionRecall, getIntersection, relativeToAbsolute, readResults

cfgpath = "yolo-obj.cfg"
weightpath = "yolo-obj_1500.weights"
objpath = "obj.data"

"""
Checks if the two classes a and b match
Returns True or False
"""
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


"""
Crops all images from the imagelist and saves the croppings in two csv files, one for fins and one for flukes.
filepath refers to the filepath of the output files without a .csv ending
"""
def cropAll(imagelist,filepath): 
  net = Detector(bytes(cfgpath, encoding="utf-8"), bytes(weightpath, encoding="utf-8"), 0, bytes(objpath,encoding="utf-8"))
  fins = []
  flukes = []
  counter = 0
  for i in imagelist:
    counter += 1
    img = cv2.imread(i)
    img_darknet = Image(img)
    results = net.detect(img_darknet)
    i_name = i.split('/')[-1]
    for cat, score, bounds in results:
      if 'fluke' in str(cat):
        x,y,w,h= bounds
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),img.shape[0])
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),img.shape[1])
        flukes.append([i_name,[x1,y1,x2,y2]])
      elif 'fin' in str(cat):
        x,y,w,h= bounds
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),img.shape[0])
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),img.shape[1])
        fins.append([i_name,[x1,y1,x2,y2]])
  with open(filepath + "fins.csv", "a") as myfile:
        wr = csv.writer(myfile)
        for r in fins:
          wr.writerow(r)
  with open(filepath + "flukes.csv","a") as myfile:
        wr = csv.writer(myfile)
        for r in flukes:
          wr.writerow(r)
  print("DONE")

"""
Crops all images in the folder. filepath refers to the filepath of the output files without a .csv ending.
"""
def cropAllInFolder(folder,filepath):
  imagelist = glob.glob(str(folder + "*.jpg"))
  imagelist.extend(glob.glob(str(folder + "*.JPG")))
  imagelist.extend(glob.glob(str(folder + "*.jpeg")))
  imagelist.extend(glob.glob(str(folder + "*.JPEG")))
  cropAll(imagelist,filepath)

"""
Evaluates a set and writes the output into a file with the filename at the given filepath.
"""
def evaluateSet(setpath,filepath,filename):
  imagelist = readSet(setpath)
  net = Detector(bytes(cfgpath, encoding="utf-8"), bytes(weightpath, encoding="utf-8"), 0, bytes(objpath,encoding="utf-8"))
  rows = []
  counter = 0
  for i in imagelist:
    counter += 1
    img = cv2.imread(i)
    img_darknet = Image(img)
    results = net.detect(img_darknet)
    for cat, score, bounds in results:
      rows.append([i,cat,score,bounds])
  with open(filepath + filename,"a") as myfile:
        wr = csv.writer(myfile)
        for r in rows:
          wr.writerow(r)  

"""
Calculates the MAP, AP50 and AP75 of a set given the ground truth setpath and the predicted resultpath
Returns the MAP overall and the AP50 and AP75
"""
def evaluateResults(setpath,resultspath):
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
                    x,y,w,h = relativeToAbsolute(l[1],l[2],l[3],l[4],img.shape[1],img.shape[0]) #label struktur original x,y,w,h in relativ #fpr the fifth time, yes this is correct shape[1] = x, shape[0] = y
                    groundtruth = [float(x-w/2),float(y-h/2),float(x+w/2),float(y+h/2)]
                    coords = [float(pred[0]-pred[2]/2),float(pred[1]-pred[3]/2),float(pred[0]+pred[2]/2),float(pred[1]+pred[3]/2)]
                    iou = getIoU(groundtruth,coords)
                    if iou > best_iou:
                        best_iou = iou
            for threshold in iouthresholds:
              if best_iou >= threshold:
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


"""
Reads in a label with the given filename
Returns only lines with content
"""
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


"""
reads in files that belong to set, and changes the paths according to the filename, e.g. filename = ../val.txt, original_path= ../obj/bla.jpg -> ../bla.jpg
"""
def readSet(filename): 
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

"""
Runs one Image through the network and saves it with the bounding box and label illustrating the output
"""
def saveOutputForImage(image,filepath): #filepath of the output image without .jpg
  net = Detector(bytes("yolo-obj.cfg", encoding="utf-8"), bytes("yolo-obj_1500.weights", encoding="utf-8"), 0, bytes("obj.data",encoding="utf-8"))
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
    cv2.rectangle(img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (255,0,0), thickness=2)
    cv2.putText(img,str(cat.decode("utf-8")),(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))
  output_name = str(filepath + "output_" + image.split('/')[-1])
  cv2.imwrite(output_name,img)

"""
Evaluates the test set and saves it accordingly
"""
def testEval():
  path = "yolo_test_results.csv"
  with open(path, "w") as myfile:
    wr = csv.writer(myfile)
    wr.writerow(["EPOCH","MAP","AP50","AP75"])
    m_ap,ap50,ap75 = evaluateResults("../data/yolo/test.txt",str("results_test.csv"))
    wr.writerow([m_ap,ap50,ap75])

"""
Evaluates all the validation sets and creates a nice, shiny graph
"""
def valEval():
  path = "yolo_eval_results.csv"
  maps = []
  ap50s = []
  ap75s = []
  with open(path, "w") as myfile:
    wr = csv.writer(myfile)
    wr.writerow(["EPOCH","MAP","AP50","AP75"])
    for i in range(600,4000,100):
      m_ap,ap50,ap75 = evaluateResults("../data/yolo/val.txt",str("yoloresults/results_" + str(i) + ".csv"))
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
  plt.savefig('yolo_eval_results.png')