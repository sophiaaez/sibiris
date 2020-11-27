from sklearn.manifold import TSNE
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def findMatches(id_,ids,cutoff=7):
    idx = []
    for i in range(len(ids)):
        if id_ == ids[i,-1] and len(idx) < cutoff:
            idx.append(i)
    return(idx)

def findIndices(tr_ids,cutoff=7):
    tr_idx = []
    i = 0
    uids = 1
    fave_whale_id = "w_b3ca4b7"
    tr_idx = findMatches(fave_whale_id,tr_ids)
    while uids <= 9:
        id_ = tr_ids[i,-1]
        if not id_ == 'new_whale' and not id_ == "":
            uids += 1
            count = 0
            for j in range(len(tr_ids)):
              if not (i == j) and tr_ids[i,-1] == tr_ids[j,-1] and count < cutoff:
                tr_idx.append(j)
                count += 1
                if count == 1:
                    tr_idx.append(i)
                    print(id_)
                    uids += 1
        i += 1
    print(len(tr_idx))
    return(tr_idx)




tr_enc = np.load("../ae/ae_training_encodings_simple.npy")
tr_ids = np.load("../ae/ae_training_ids_simple.npy")
te_enc = np.load("../ae/ae_test_encodings_simple.npy")
te_ids = np.load("../ae/ae_test_ids_simple.npy")
ids =np.concatenate([tr_ids,te_ids],axis=0)
enc =np.concatenate([tr_enc,te_enc],axis=0)
idx = findIndices(ids)
ids = ids[idx]
enc = enc[idx]

tsne = TSNE(n_components=2)
transformed = tsne.fit_transform(enc)
print(transformed.shape)

font = {'family':'normal',
        'size':4}
matplotlib.rc('font',**font)

fig, ax = plt.subplots(dpi=300)
plt.scatter(transformed[:,0],transformed[:,1])
for i, txt in enumerate(ids):
    ax.annotate(str(txt[0]+"\n"+ txt[-1]),(transformed[i,0],transformed[i,1]),)
    plt.savefig('analysis_plot.png')