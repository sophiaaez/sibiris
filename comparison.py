#connection to database
from database import *
from sklearn.neighbors import NearestNeighbors

def findClosestFluke(fluke,n):
    results = []
    embed = fluke[2] #fluke is strctured [imagepath,middlepoint,embedding]
    knut = KNN(database.x(),database.y(),k=n)
    closest = knut.predictTopK(embed) #now we got imagpaths/names
    for c in closest:
        c_id = database.getIDbyImagepath(imagepath)


    return closest

class KNN:
    def __init__(self,x,y,k=10):
        self.neighbourhood = NearestNeighbors(n_neighbors=k,algorithm='brute')
        self.neighbourhood.fit(x)
        #self.embeddings = x
        self.ids = y #ids refers to the image name not individual name 
        
    def predictTopK(self,x): #returns an array of the top k neighbours (imagepaths/names) and their distance 
        distances, neigh_idxs = self.neighbourhood.kneighbors([x])
        distances = distances[0] 
        neigh_idxs = neigh_idxs[0]
        for n in range(len(neigh_idxs)):
            idx = neigh_idxs[n]
            results.append([self.ids[idx],distances[n]])
        return results
    
    def retrain(self,x,y):
        self.neighbourhood.fit(x)
        self.ids = y