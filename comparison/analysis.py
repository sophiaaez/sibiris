from sklearn.manifold import TSNE
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


"""
finds matches for an id_ among a list ids, the amount of matches is limited by cutoff
Returns a list of indices that refer to the id_ matches position in ids.
"""
def findMatches(id_,ids,cutoff=7):
    idx = []
    for i in range(len(ids)):
        if id_ == ids[i,-1] and len(idx) < cutoff:
            idx.append(i)
    return(idx)

"""
Finds a certain amonut of indexes of ids that appear multiple times,
limits the number of the same id appearing there by the cutoff
Returns a list of indices of those images/ids
"""
def findIndices(tr_ids,idlist,amount = 9,cutoff=7):
    tr_idx = []
    if idlist:
        for idl in idlist:
            print(idl)
            tr = findMatches(idl,tr_ids)
            tr_idx.extend(tr)
    else:
        i = 0
        uids = 1
        fave_whale_id = "w_b3ca4b7"
        tr_idx = findMatches(fave_whale_id,tr_ids)
        while uids <= amount:
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
    return(tr_idx)

"""
Creates a plot of the encodings at encodingspath whose ids are at idspath. both should have .npy format.
Saves the plot at plotpath should include .png or .jpg ending
"""
def createPlot(encodingpath,idspath,plotpath,idlist=None):
    enc = np.load(encodingpath)
    ids = np.load(idspath)
    idx = findIndices(ids,idlist)
    ids = ids[idx]
    enc = enc[idx]
    tsne = TSNE(n_components=2)
    transformed = tsne.fit_transform(enc)
    font = {'family':'normal',
            'size':4}
    matplotlib.rc('font',**font)
    fig, ax = plt.subplots(dpi=300)
    plt.scatter(transformed[:,0],transformed[:,1])
    for i, txt in enumerate(ids):
        ax.annotate(str(txt[0]+"\n"+ txt[-1]),(transformed[i,0],transformed[i,1]),)
        plt.savefig(plotpath)

