import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score,recall_score
import torch.nn.functional as F
import math

def gaussian(ins, is_training, stddev=0.2):
    if is_training:
        return ins + Variable(torch.randn(ins.size()).cuda() * stddev)
    return ins

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self,filters):
        super(UnFlatten,self).__init__()
        self.filters = filters
    def forward(self, input):
        isize = int(math.sqrt(input.size(1)/self.filters))
        return input.view(input.size(0),self.filters, isize, isize) #4,128,128 

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork,self).__init__()
        self.uf = UnFlatten(32)
        self.l1 = nn.Conv2d(32, 64, 3, stride=1,padding=1)
        self.l2 = nn.Conv2d(64, 64, 3, stride=2,padding=2)
        self.l3 = nn.Conv2d(32, 64, 3, stride=2,padding=2)
        self.b1d32 = nn.BatchNorm1d(32)
        self.bn = nn.BatchNorm2d(64)
        self.bnn = nn.BatchNorm2d(128)
        self.bn1d = nn.BatchNorm1d(128)

        self.l11 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
        self.l22 = nn.Conv2d(64, 64, 3, stride=2,padding=2)
        self.l33 = nn.Conv2d(64, 64, 3, stride=2,padding=2) 

        self.l111 = nn.Conv2d(64, 128, 3, stride=1,padding=1)
        self.l222 = nn.Conv2d(128, 128, 3, stride=2,padding=2)
        self.l333 = nn.Conv2d(64, 128, 3, stride=2,padding=2) 

        self.o = nn.Conv2d(128,256,1,stride=1,padding=1)
        self.fl = Flatten()
        self.d = nn.Linear(16384,512)

        self.c1 = nn.Linear(1024,128)
        self.c2 = nn.Linear(128,32)
        self.c3 = nn.Linear(32,1)


    def forward_once(self, x):
          # Forward pass 
          #gaussian noise
          #print(x.size())
          x = self.uf(x)
          x1 = nn.ReLU()(self.l1(x))
          x2 = nn.ReLU()(self.l2(x1))
          x3 = self.l3(x)
          x = nn.ReLU()(x2 + x3)
          x = self.bn(x)
          #print(x.size())

          x1 = nn.ReLU()(self.l11(x))
          x2 = self.l22(x1)
          x3 = self.l33(x)
          x = nn.ReLU()(x2 + x3)
          x = self.bn(x)
          #print(x.size())

          x1 = nn.ReLU()(self.l111(x))
          x2 = self.l222(x1)
          x3 = self.l333(x)
          x = nn.ReLU()(x2 + x3)
          x = self.bnn(x)
          #print(x.size())

          x = self.o(x)
          #print(x.size())
          x = self.fl(x)
          #print(x.size())
          x = nn.ReLU()(self.d(x))
          #print(x.size())
          return x

    def forward(self,input1,input2):
        o1 = self.forward_once(input1)
        o2 = self.forward_once(input2)
        #print("forwarded")
        x = torch.cat((o1,o2),axis=-1)
        #print(x.size())
        x = nn.ReLU()(self.c1(x))
        #print(x.size())
        x = self.bn1d(x)
        x = nn.Dropout2d(0.25)(x)
        x = nn.ReLU()(self.c2(x))
        #print(x.size())
        x = self.b1d32(x)
        x = nn.Sigmoid()(self.c3(x))

        return(x)



def createBatch(batch_size,enc,ids):
  def find_same(id_):
    for i in range(len(ids)):
      if  ids[i,-1] == id_:
        return i
    return -1
  #batch = []
  b1 = []
  b2 = []
  labels = []
  for i in range(batch_size):
    i1 = np.random.randint(0,len(ids))
    if i % 3 == 0:
      i2 = find_same(ids[i1,-1])
    else:
      i2 = np.random.randint(0,len(ids))
    o = 0
    if ids[i1,-1] == ids[i2,-1]:
      o = 1
    b1.append(enc[i1])
    b2.append(enc[i2])
    #batch.append([enc[i1],enc[i2]])
    labels.append([o])
  return (torch.from_numpy(np.array(b1)).float().cuda(),torch.from_numpy(np.array(b2)).float().cuda(),torch.from_numpy(np.array(labels)).float().cuda())

class EarlyStopper(): #patience is amount of validations to wait without loss improvment before stopping, delta is the necessary amount of improvement
    def __init__(self,patience,delta,save_path,save):
        self.patience = patience
        self.patience_counter = 0
        self.delta = delta
        self.save_path = save_path
        self.best_loss = -1
        self.save = save

    def earlyStopping(self,loss,model):
        if self.best_loss == -1 or loss < self.best_loss - self.delta : #case loss decreases
            self.best_loss = loss
            if self.save:
                torch.save(model,str(self.save_path))
            self.patience_counter = 0
            return False
        else: #case loss remains the same or increases
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True


def train(epochs,learning_rate,batch_size,set1,ids1):
    model = SiameseNetwork().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    no_improve = 0
    last_loss = 10000000
    model.train()
    mse = nn.MSELoss()
    es = EarlyStopper(patience=50,delta=0.001,save_path="siamese_network_vae_",save=True)
    for epoch in range(epochs):
        total_train_loss = 0
        optimizer.zero_grad()
        labels = []
        predictions = []
        for i in range(int(np.ceil(len(set1)/batch_size))):
            b1,b2,l = createBatch(batch_size,set1,ids1)
            o = model(b1,b2)
            loss = mse(o,l)
            loss.backward()
            optimizer.step()
            total_train_loss += loss
            labels.extend(l[:,0].tolist())
            predictions.extend(o[:,0].tolist())
        a = accuracy_score(labels,np.where(np.array(predictions) < 0.5, 0.0,1.0))
        r = recall_score(labels,np.where(np.array(predictions) < 0.5, 0.0,1.0))
        print("EPOCH " + str(epoch) + " with loss "  + str(total_train_loss.item()) + ", accuracy " + str(a) + " and recall " + str(r))
        stop = es.earlyStopping(total_train_loss,model)
        if stop:
          print("TRAINING FINISHED AFTER " + str(epoch-50) + " EPOCHS. K BYE.")
          break

def findIndices(tr_ids,size=500,cutoff=3):
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
    print(len(tr_idx))
    return(tr_idx)

def trial():
    tr_enc = np.load("../ae/vae_training_encodings.npy")
    tr_ids = np.load("../ae/vae_training_encoding_ids.npy")
    tr_idx = findIndices(tr_ids)
    xs_tr_enc = tr_enc[tr_idx]
    xs_tr_ids = tr_ids[tr_idx]
    train(350,0.0001,64,xs_tr_enc,xs_tr_ids)


def matchTop10():
    model = torch.load("siamese_network_vae_").cuda()
    tr_enc = np.load("../ae/vae_training_encodings.npy")
    tr_ids = np.load("../ae/vae_training_encoding_ids.npy")
    te_enc = np.load("../ae/vae_test_encodings.npy")
    te_ids = np.load("../ae/vae_test_encoding_ids.npy")
    tr_idx = findIndices(tr_ids)
    tr_enc = tr_enc[tr_idx]
    tr_ids = tr_ids[tr_idx]
    te_idx = findIndices(te_ids)
    te_enc = te_enc[te_idx]
    te_ids = te_ids[te_idx]
    pred = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(te_ids)):
      te = torch.from_numpy(np.array([te_enc[i],te_enc[i],te_enc[i],te_enc[i],te_enc[i]])).float().cuda()
      matches = []
      for j in range(0,len(tr_ids)-5,5):
        tr = torch.from_numpy(np.array([tr_enc[j],tr_enc[j+1],tr_enc[j+2],tr_enc[j+3],tr_enc[j+4]])).float().cuda()
        res = model(te,tr)
        for r in res:
          if r > 0.5: #classified match 
            matches.append([r.item(),tr_ids[j,-1]])
            if te_ids[i,-1] == tr_ids[j,-1]: #and is match
              tp += 1
            else: #but is no match
              fp += 1
          else: #classified not match
            if te_ids[i,-1] == tr_ids[j,-1]: #but is match
              fn += 1
            else: #and is no match
              tn += 1
      matches = np.array(matches)
      matches = matches[matches[:,0].argsort()]
      match = 0
      for k in range(1,11):
        m_id = matches[-k,-1]
        if te_ids[i,-1] == m_id:
          match = k
          break
      if match > 0:
        pred.append(1)
      else: 
        pred.append(0)
      if tp+fp > 0 and tp+fn > 0:
        print("recall: " + str(tp/(tp+fn)) + " and precision: " + str(tp/(tp+fp)))
    a = np.mean(np.array(pred))
    print("test accuracy: " + str(a))
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    print("recall: " + str(recall) + " and precision: " + str(precision))


torch.cuda.empty_cache()
trial()
#train(350,0.0001,64,np.load("../ae/ae_training_encodings.npy"),np.load("../ae/ae_training_encoding_ids.npy"))
matchTop10()