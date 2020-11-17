from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np


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
                    results.append([self.idsdis[idx],distances[n]])
            else:
                break

        return results
    
    def retrain(self,x,y):
        self.neighbourhood.fit(x)
        self.ids = y

def evalSet(database_encodings, database_ids,set_encodings,set_ids,importance=None):
    encodings = np.load(database_encodings)#"../ae/encodings.npy")
    encoding_ids = np.load(database_ids, allow_pickle=True)#"../ae/encoding_ids.npy")
    test_encodings = np.load(set_encodings)
    test_ids = np.load(set_ids, allow_pickle=True)
    if importance:
        important_dimensions = np.mean(np.load(importance),axis=0)
        encodings = encodings * important_dimensions
        test_encodings = test_encodings * important_dimensions
    knut = KNN(encodings,encoding_ids,10)
    t1 = 0
    t5 = 0
    t10 = 0
    matchlist = []
    matchables = len(test_encodings)
    match_dist = []
    for ite in range(len(test_encodings)):
        if ite%1000 == 0:
            print(ite)
        results = knut.predictTopK(test_encodings[ite]) #results are structured [[name,bbox,label],distance]
        #print(test_ids[ite])
        if test_ids[ite][2] == "new_whale":
            matchables -= 1
        for ir in range(len(results)):
            #print(results[ir])
            if len(results[ir]) > 2:# and not results[ir][2] == "new_whale" and not results[ir][2] == "":
                if results[ir][2] == test_ids[ite]: #if the class in the results matches the one this encoding belongs to
                    match_dist.append(results[ir][-1])
                    if ir == 0:
                        t1 += 1
                    if ir < 5:
                        t5 += 1
                    t10 += 1
                    print("MATCH")
                    matchlist.append(test_ids[ite])
    print("TOP 1: " + str(t1/matchables))
    print("TOP 5: " + str(t5/matchables))
    print("TOP 10: "  + str(t10/matchables))
    print(matchlist)


def fixSet(encodings_path,name):
    new_encodings = []
    encodings = np.load(encodings_path)
    for e in range(len(encodings)):
        new_encodings.append(encodings[e][0])
    with open(str(name) + '.npy', 'wb') as f:
        np.save(f, new_encodings)

def fixLabels(labels_path,name):
    new_labels = []
    labels = np.load(labels_path,allow_pickle=True)
    print(labels.shape)
    label = [labels[0]]
    for l in range(1,len(labels)):
        if ".jpg" in str(labels[l]):
            new_labels.append(label)
            label = []
        label.append(str(labels[l]))

    new_labels.append(label)
    new_numpy = np.array(new_labels)
    print(new_numpy.shape)
    with open(str(name) + '.npy', 'wb') as f:
        np.save(f, new_numpy)

def checkDist(database_encodings, database_ids,set_encodings,set_ids,whale_id):
    encodings = np.load(database_encodings)#"../ae/encodings.npy")
    encoding_ids = np.load(database_ids, allow_pickle=True)#"../ae/encoding_ids.npy")
    test_encodings = np.load(set_encodings)
    test_ids = np.load(set_ids, allow_pickle=True)
    knut = KNN(encodings,encoding_ids,10)

    index1 = np.where(test_ids == whale_id)
    index2 = np.where(encoding_ids == whale_id)
    match_encs = []
    for i1 in index1[0]:
        match_encs.append(test_encodings[i1])
    for i2 in index2[0]:
        match_encs.append(encodings[i2])
    match_dist = np.zeros((len(match_encs),len(match_encs)))
    for m in range(len(match_encs)):
        for n in range(len(match_encs)):
            match_dist[m,n] = np.linalg.norm(match_encs[m]-match_encs[n])
    print(match_dist)


def analyseDimensions(database_encodings,database_ids):
    encodings = np.load(database_encodings)#"../ae/encodings.npy")
    encoding_ids = np.load(database_ids, allow_pickle=True)#"../ae/encoding_ids.npy")
    ids = encoding_ids[:,2] #only actual ids
    #clean ids from "new_whale" and ""
    remove_idx = []
    for i in range(len(ids)):
        if ids[i] == "" or ids[i] == "new_whale":
            remove_idx.append(i)
    encodings = np.delete(encodings,remove_idx, axis=0)
    ids = np.delete(ids,remove_idx, axis=0)
    #create classifier
    classifier = LDA() #or Perceptron()
    classifier.fit(encodings,ids)
    importance = classifier.coef_
    np.save("vae_importance.npy",importance)
    #return importance

def tsnegedoens(encodings,encoding_ids):
    encode = np.load(encodings)
    ids = np.laod(encoding_ids)
    #


#fixLabels("../ae/encoding_ids.npy","../ae/encoding_ids_simple")
#fixSet("../ae/test_encodings.npy","../ae/test_encodings_simple")
#evalSet("../ae/ae_training_encodings.npy","../ae/ae_training_encoding_ids.npy","../ae/ae_test_encodings.npy","../ae/ae_test_encoding_ids.npy")#,importance="./importance.npy")
#evalSet("../ae/vae_training_encodings.npy","../ae/vae_training_encoding_ids.npy","../ae/vae_test_encodings.npy","../ae/vae_test_encoding_ids.npy")#,importance="./importance.npy")

#analyseDimensions("../ae/vae_training_encodings.npy","../ae/vae_training_encoding_ids.npy")
#evalSet("../ae/vae_training_encodings.npy","../ae/vae_training_encoding_ids.npy","../ae/vae_test_encodings.npy","../ae/vae_test_encoding_ids.npy",importance="./vae_importance.npy")
#checkDist("../ae/ae_encodings.npy","../ae/ae_encoding_ids.npy","../ae/ae_test_encodings.npy","../ae/ae_test_encoding_ids.npy", "w_b3ca4b7")
