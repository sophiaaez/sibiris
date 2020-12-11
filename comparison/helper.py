import numpy as np

"""
Finds the indices of all unlabelled or 'new_whale' data and
Returns the indices of those as a list
"""
def cleanDataset(tr_ids):
    tr_idx= []
    for i in range(len(tr_ids)):
        if not (tr_ids[i,-1] == "new_whale") and not (tr_ids[i,-1] == ""):
            tr_idx.append(i)
    return tr_idx


def randomSubset(size,len_set):
  idx = []
  while len(idx) < size:
    new_idx = np.random.randint(0,len_set)
    if not new_idx in idx:
      idx.append(new_idx)
  return idx
