# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:34:23 2020

@author: monaw
"""

import csv
import glob
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time

class Labels:
    def __init__(self, path):
        self.labels = self.prepareLabels(path)

    def prepareLabels(self,path):
        #read in all
        labels = []
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                r = ''.join(row) 
                labels.append(r)  
        labels.remove(labels[0]) #delete headline [Image, Id]
        return labels
    
    def get(self):
        return self.labels
    
    def findImageById(self,id):
        matches = [x for x in self.labels if id in x]
        length = len(id) +1 #+1 for the separating comma
        for m in range(len(matches)):
            matches[m] = matches[m][:-length]
        return matches
        
    def findIdByImage(self,image):
        matches = [x for x in self.labels if image in x]
        length = len(image) +1 #+1 for the separating comma
        for m in range(len(matches)):
            matches[m] = matches[m][length:]
        return matches
    
    def labelImages(self,images):
        ls = [None] * len(images)
        for i in range(len(images)):
            ls[i] = self.findIdByImage(images[i])[0]
        return ls

class KNN:
    def __init__(self,x,y,k=10):
        self.neighbourhood = NearestNeighbors(n_neighbors=k,algorithm='brute')
        self.neighbourhood.fit(x)
        #self.embeddings = x
        self.ids = y #ids refers to the image name not individual name 
        self.labels=Labels('../data/train.csv')
        
    def predictTopK(self,x): #returns an array of the top k neighbours and their distance
        distances, neigh_idxs = self.neighbourhood.kneighbors([x])
        distances = distances[0] 
        neigh_idxs = neigh_idxs[0]
        results=[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        for n in range(len(neigh_idxs)):
            idx = neigh_idxs[n]
            results[n][0] = self.ids[idx]
            results[n][1] = self.labels.findIdByImage(self.ids[idx].split('/')[-1])
            results[n][2] = distances[n]
        return results
    
    def retrain(self,x,y):
        self.neighbourhood.fit(x)
        self.ids = y

def flattenThisBS(bs):
    newbs = []
    for b in range(len(bs)):
        if b == 0:
            newbs = np.array([bs[b].flatten()])
        else:
            newbs = np.append(newbs,[bs[b].flatten()],axis=0)
    return newbs

def fakeIdBS(bspath):
    imagelist = glob.glob(bspath + str("*.jpg"))
    return np.array(imagelist)


#l = Labels('DATA/Kaggle/train.csv')
#labels = l.get()
enc = np.load("../ae/encodings.npy")
print(enc.shape)
#enc = flattenThisBS(enc)
enc = enc.reshape(enc.shape[0],enc.shape[1]*enc.shape[2]*enc.shape[3])
print(enc.shape)
enc_ids = fakeIdBS("../data/train/crops/")
print(enc_ids.shape)
start = time.time()
k = KNN(enc,enc_ids)
print("Creating nn took {:.2f}s (aka forever)".format(time.time() - start))
start = time.time()
k.predictTopK(enc[0])
print("Search took {:.2f}s (aka forever)".format(time.time() - start))