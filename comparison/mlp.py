from sklearn.neural_network import MLPClassifier
import numpy as np
from helper import cleanDataset,randomSubset


"""
Creates and trains an MLP with the given hidden_layer_sizes (single integer or tuple of ints)
using the activation function activation
on the data set tr_enc_path and tr_ids_path (both .npy files) 
Returns (and prints) the top 10 accuracy based on the training set te_enc_path and te_ids_path (both .npy files)
"""
def top10MLP(hidden_layer_sizes,tr_enc_path,tr_ids_path,te_enc_path,te_ids_path,activation="relu",reduced=None):
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
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,activation = activation, max_iter=1000)
    mlp.fit(tr_enc,tr_ids[:,-1])
    top10matches = []
    c = mlp.classes_
    for x in range(len(te_enc)):
        pp = mlp.predict_proba([te_enc[x]])[0]
        top10cidx = []
        for i in range(10):
            max_ = 0
            maxidx = 0
            for j in range(len(pp)):
                if max_ < pp[j] and not (j in top10cidx):
                    max_ = pp[j]
                    maxidx = j
            top10cidx.append(maxidx)
        top10c = c[top10cidx]
        if te_ids[x,-1] in top10c:
            top10matches.append(te_ids[x,-1])
    print(len(top10matches)/len(te_ids))
    return(len(top10matches)/len(te_ids))
