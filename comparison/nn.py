from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from helper import cleanDataset,randomSubset


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
        
    def predictTopK(self,x): #returns an array of the top k neighbours (imagepaths/bbox/label) and their distance 
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

"""
Evaluates a dataset for the ModNN above 'trained' on the data set tr_enc_path and tr_ids_path (both .npy files)
evalutes based on the set_encodings/_ids, if only a partial evaluation is wanted, reduced can be set to an integer/cutoff point
Returns (and prints) the top 10 accuracy based on the training set te_enc_path and te_ids_path (both .npy files) 
"""
def top10NN(tr_enc_path,tr_ids_path,te_enc_path,te_ids_path,reduced=None):
    tr_enc = np.load(tr_enc_path)
    tr_ids = np.load(tr_ids_path)
    te_enc = np.load(te_enc_path)
    te_ids = np.load(te_ids_path)
    tr_idx = cleanDataset(tr_ids)
    tr_enc = tr_enc[tr_idx]
    tr_ids = tr_ids[tr_idx]
    te_idx = cleanDataset(te_ids)
    te_enc = te_enc[te_idx]
    te_ids = te_ids[te_idx]
    if reduced:
        te_idx = randomSubset(reduced,len(te_ids))
        te_enc = te_enc[te_idx]
        te_ids = te_ids[te_idx]
    knut = ModNN(tr_enc,tr_ids,10)
    t10 = 0
    matchlist = []
    matchables = len(te_enc)
    for ite in range(len(te_enc)):
        results = knut.predictTopK(te_enc[ite]) #results are structured [[name,bbox,label],distance]
        for ir in range(len(results)):
            if results[ir][0][-1] == te_ids[ite,-1]: #if the class in the results matches the one this encoding belongs to
                    t10 += 1
                    matchlist.append(results[ir])
                    break
    print("TOP 10: "  + str(t10/matchables))
    #np.save("matches_nn.npy",np.array(matchlist))
    return(t10/matchables)