from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
from matplotlib import pyplot as plt
import os



"""
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
xs_te_ids = te_ids[te_idx]"""

"""tsne = TSNE(n_components=2)
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
    os.system(str("cp ../data/kaggle/" + str(i) + " imgs/"))"""

def findIndices(tr_ids,size=500,cutoff=5):
    tr_idx = []
    i = 0
    while len(tr_idx)<=500:
        id_ = tr_ids[i,-1]
        tr_idx.append(i)
        count = 0
        for j in range(len(tr_ids)):
          if not (i == j) and tr_ids[i,-1] == tr_ids[j,-1]:
            tr_idx.append(j)
            count += 1
            if count >= cutoff:
              break
        i += 1
    #print(len(tr_idx))
    return(tr_idx)

def findMatchingIndices(te_ids,tr_ids):
    unique_ids = set(tr_ids[:,-1])
    te_idx = []
    for u in unique_ids:
        for t in range(len(te_ids)):
            if u == te_ids[t,-1]:
                te_idx.append(t)
    #print(len(te_idx))
    return te_idx


def cleanDataset(tr_ids):
    tr_idx= []
    for i in range(len(tr_ids)):
        if not (tr_ids[i,-1] == "new_whale") and not (tr_ids[i,-1] == ""):
            tr_idx.append(i)
    print(len(tr_idx))
    return tr_idx

"""
tr_enc = np.load("../ae/ae_training_encodings.npy")
tr_ids = np.load("../ae/ae_training_encoding_ids.npy")
te_enc = np.load("../ae/ae_test_encodings.npy")
te_ids = np.load("../ae/ae_test_encoding_ids.npy")
tr_idx = findIndices(tr_ids)
tr_enc = tr_enc[tr_idx]
tr_ids = tr_ids[tr_idx]
te_idx = findMatchingIndices(te_ids,tr_ids)
te_enc = te_enc[te_idx]
te_ids = te_ids[te_idx]
"""


#mlp = MLPClassifier(hidden_layer_sizes=(400, 100,50),activation = 'relu',random_state=1, max_iter=1000)
#mlp.fit(tr_enc,tr_ids[:,-1])
"""
print(te_ids[0])
p = mlp.predict([te_enc[0]])
print(str(p))
print(len(p))
pp = mlp.predict_proba([te_enc[0]])
print(str(pp))"""
#s = mlp.score(te_enc,te_ids[:,-1])
#print(s)

#pred = mlp.predict(te_enc)
#print(pred)
#a_s = accuracy_score(pred,te_ids[:,-1],normalize=True)
#print("MLP ACCURACY" + str(a_s))
"""
svc = SVC()
svc.fit(xs_tr_enc,xs_tr_ids[:,-1])
pred = svc.predict(xs_te_enc)
#print(pred)
a_s = accuracy_score(pred,xs_te_ids[:,-1],normalize=True)
print("SVC ACCURACY" + str(a_s))"""

def top10MLP(hidden_layer_sizes,activation,reduced=False):
    tr_enc = np.load("../ae/ae_training_encodings_simple.npy")
    tr_ids = np.load("../ae/ae_training_ids_simple.npy")
    te_enc = np.load("../ae/ae_test_encodings_simple.npy")
    te_ids = np.load("../ae/ae_test_ids_simple.npy")
    tr_idx = cleanDataset(tr_ids)
    tr_enc = tr_enc[tr_idx]
    tr_ids = tr_ids[tr_idx]
    te_idx = cleanDataset(te_ids)
    te_enc = te_enc[te_idx]
    te_ids = te_ids[te_idx]
    if reduced:
        tr_idx = findIndices(tr_ids)
        tr_enc = tr_enc[tr_idx]
        tr_ids = tr_ids[tr_idx]
        te_idx = findMatchingIndices(te_ids,tr_ids)
        te_enc = te_enc[te_idx]
        te_ids = te_ids[te_idx]
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,activation = activation,random_state=1, max_iter=1000)
    mlp.fit(tr_enc,tr_ids[:,-1])
    top10matches = []
    c = mlp.classes_
    for x in range(len(te_enc)):
        t = te_ids[x]
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
            pp[maxidx] = 0
        top10c = c[top10cidx]
        if t[-1] in top10c:
            top10matches.append(t)
    print(len(top10matches)/len(te_ids))
    return(len(top10matches)/len(te_ids))

print("100 relu")
top10MLP((100),"relu",reduced=False)
#print("100 logistic")
#top10MLP((100),"logistic",reduced=False)
print("200 relu")
top10MLP((200),"relu",reduced=False)
#print("200 logistic")
#top10MLP((200),"logistic")
print("100,50 relu")
top10MLP((100,50),"relu",reduced=False)
#print("100,50 logistic")
#top10MLP((100,50),"logistic")
print("200,100 relu")
top10MLP((200,100),"relu",reduced=False)
#print("200,100 logistic")
#top10MLP((200,100),"logistic")
print("50 relu")
top10MLP((50),"relu",reduced=False)
#print("200 logistic")
#top10MLP((200),"logistic",reduced=False)
print("(400) relu")
top10MLP((400),"logistic",reduced=False)