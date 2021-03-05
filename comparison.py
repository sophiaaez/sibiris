#connection to database
from sklearn.neighbors import NearestNeighbors
import numpy as np

database_encodings = "./ae/ae_training_encodings_simple_v3.npy"
database_ids = "./ae/ae_training_ids_simple_v3.npy"

def findClosestFluke(fluke,n=10): #flukes is the embedding 
    results = []
    embed = fluke[1] 
    database_x = np.load(database_encodings)
    database_y = np.load(database_ids)
    knut = KNN(database_x,database_y,k=n)
    closest = knut.predictTopK(embed) 
    print(closest)
    return closest

"""
ModNN, a modified version of the nearest neighbour algorithm using brute force
predictTopK finds the k closest matches
"""
class ModNN:
    def __init__(self,embeddings,labels,k=10):
        self.neighbourhood = NearestNeighbors(n_neighbors=k*10,algorithm='brute')
        self.neighbourhood.fit(embeddings)
        self.truek = k
        self.ids = labels #ids refers to [imagepath,bounding box,individual label] 
        #embeddings and labels have to be orderd the same for this to work!!!
        
    def predictTopK(self,x): #returns an array of the top k neighbours (imagepaths/bbox/label) with unique labels and their distance 
        #print(x.shape)
        distances, neigh_idxs = self.neighbourhood.kneighbors([x])
        distances = distances[0] #sorted in ascending order
        neigh_idxs = neigh_idxs[0]
        results = []
        individual_ids = []
        for n in range(len(neigh_idxs)):
            idx = neigh_idxs[n]
            if len(results) < self.truek:
                if not self.ids[idx][-1] in individual_ids:
                    individual_ids.append(self.ids[idx][-1])
                    results.append([self.ids[idx],distances[n]])
            else:
                break
        return results
    
    def retrain(self,x,y):
        self.neighbourhood.fit(x)
        self.ids = y