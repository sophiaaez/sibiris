from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
from matplotlib import pyplot as plt
import os


tr_enc = np.load("../ae/vae_training_encodings.npy")
tr_ids = np.load("../ae/vae_training_encoding_ids.npy")
te_enc = np.load("../ae/vae_test_encodings.npy")
te_ids = np.load("../ae/vae_test_encoding_ids.npy")

def findIndices(tr_ids,te_ids,size=500,cutoff=10):
    tr_idx = []
    te_idx = []
    for i in range(size):
        id_ = te_ids[i,-1]
        te_idx.append(i)
        count = 0
        for j in range(len(tr_ids)):
            if id_ == tr_ids[j,-1]:
                tr_idx.append(j)
                count += 1
            if count == cutoff:
                break
    print(len(tr_idx))
    print(len(te_idx))
    return(tr_idx,te_idx)

tr_idx,te_idx = findIndices(tr_ids,te_ids,10)

xs_tr_enc = tr_enc[tr_idx]#np.concatenate([tr_enc[tr_idx],tr_enc[-200:]],axis=0)
xs_tr_ids = tr_ids[tr_idx]#np.concatenate([tr_ids[tr_idx],tr_ids[-200:]],axis=0)
xs_te_enc = te_enc[te_idx]
xs_te_ids = te_ids[te_idx]

tsne = TSNE(n_components=2)
t_encs = np.concatenate([xs_tr_enc,xs_te_enc])
t_ids = np.concatenate([xs_tr_ids,xs_te_ids])
transformed = tsne.fit_transform(t_encs)
print(transformed.shape)
font = {'family':'normal',
        'size' : 4}
matplotlib.rc('font',**font)
fig,ax = plt.subplots(dpi=300)
plt.scatter(transformed[:,0],transformed[:,1])
for i,txt in  enumerate(t_ids):
    ax.annotate(str(txt[0] + " " + txt[-1]),(transformed[i,0],transformed[i,1]),)

plt.savefig("plot1.png")

imagelist = t_ids[:,0]
for i in imagelist:
    os.system(str("cp ../data/kaggle/" + str(i) + " imgs/"))


"""
mlp = MLPClassifier(hidden_layer_sizes=(400, 100,50),random_state=1, max_iter=1000)
mlp.fit(xs_tr_enc,xs_tr_ids[:,-1])
pred = mlp.predict(xs_te_enc)
#print(pred)
a_s = accuracy_score(pred,xs_te_ids[:,-1],normalize=True)
print("MLP ACCURACY" + str(a_s))

svc = SVC()
svc.fit(xs_tr_enc,xs_tr_ids[:,-1])
pred = svc.predict(xs_te_enc)
#print(pred)
a_s = accuracy_score(pred,xs_te_ids[:,-1],normalize=True)
print("SVC ACCURACY" + str(a_s))"""