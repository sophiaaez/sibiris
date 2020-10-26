#connection to database
from database import *
from sklearn.neighbors import NearestNeighbors

"""def findClosestFluke(fluke,n): #flukes is the embedding and l
    results = []
    embed = fluke[2] 
    knut = KNN(database.x(),database.y(),k=n)
    closest = knut.predictTopK(embed) #now we got imagpaths/bbox/labels/distances
    return closest"""

class KNN:
    def __init__(self,embeddings,labels,k=10):
        self.neighbourhood = NearestNeighbors(n_neighbors=k*2,algorithm='brute')
        self.neighbourhood.fit(embeddings)
        self.truek = k
        self.ids = labels #ids refers to [imagepath,bounding box,individual label] 
        #embeddings and labels have to be orderd the same for this to work!!!
        
    def predictTopK(self,x): #returns an array of the top k neighbours (imagepaths/bbox/label) and their distance 
        distances, neigh_idxs = self.neighbourhood.kneighbors([x])
        distances = distances[0] 
        neigh_idxs = neigh_idxs[0]
        results = []
        individual_ids = []
        for n in range(len(neigh_idxs)):
            idx = neigh_idxs[n]
            if len(results) <= self.truek:
                if len(self.ids[idx]) > 2:
                    if not self.ids[idx][2] in individual_ids:
                        individual_ids.append(self.ids[idx][2])
                        results.append([self.ids[idx],distances[n]])
                else:
                    results.append([self.idx[idx],distnaces[n]])
            else:
                break
        return results
    
    def retrain(self,x,y):
        self.neighbourhood.fit(x)
        self.ids = y