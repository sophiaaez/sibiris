from pydarknet import Detector,Image
import cv2
import numpy as np
import glob
import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
from skimage import io, util, transform, color,exposure,filters


cfgpath = "yolo/yolo-obj.cfg"
weightpath = "yolo/yolo-obj_1500.weights"
objpath = "yolo/obj.data"
ae_path = "ae/VAE_earlystopsave_simple_siamese_v3.pth"
img_size = 512
database_encodings = "ae/vae_training_siamese_encodings_simple_v3.npy"
database_ids = "ae/vae_training_siamese_ids_simple_v3.npy"

class Sibiris():
    def __init__(self):
        self.fins = []
        self.flukes = []
        self.flukeembeddings = []
        self.database_x = np.load(database_encodings)
        self.database_y = np.load(database_ids)
        self.model = torch.load(ae_path)
        self.model.eval()

    def processImages(self,imagelist):
        print(imagelist)
        #preprocess them
        self.fins, self.flukes = self.preprocessImages(imagelist) #this actually works
        #send through to embedding
        self.flukeembeddings = self.embedFlukes(self.flukes)
        #find closest but loop coz user interaction iterative
        results = []
        for f in self.flukeembeddings:
            closest = self.findClosestFlukes(f,10)
            #decipher the label = [imagepath, bounding box, individual id] and distances
            #show cropped images with imagepath, individual id and distance
            #get to user input
            results.append(closest)
        print("done")
        return(results)  
    
    def preprocessImages(self,imagelist):
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
        
    def embedFlukes(self,flukes):
        embeddings = []
        for f in flukes:
            i = f[0] #flukes is structured [path,[bounding box]]
            x1,y1,x2,y2 = f[1]
            print(i)
            #load image
            image = io.imread(str(i),as_gray=True)
            image = image[y1:y2,x1:x2] #CROPPING
            image = transform.resize(image,(512,512))
            #do embedding
            _,embed,_ = self.model.encode(image)
            embeddings.append([i,embed])
        return embeddings
                
    def findClosestFlukes(self,fluke,n):
        embed = fluke[1] 
        train_enc = np.load(database_encodings)
        train_ids = np.load(database_ids)
        tr_len = len(train_ids)
        for j in range(tr_len):
            tr = train_enc[j]
            _,_,tr_id = train_ids[j]
            tr = torch.from_numpy(tr).float().cuda()
            r = self.model.siamese(embed,tr,batched=False)
            r = r[0]
            matches.append([r.item(),tr_id])
        matches = np.array(matches)
        if len(matches) > 0:
            matches = matches[matches[:,0].argsort()] #sorts from low numbers to high numbers 
        closest = matches[:n]
        return closest
          
        
