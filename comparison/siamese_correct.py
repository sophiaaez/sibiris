import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score,recall_score
import torch.nn.functional as F
import math
import optuna

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
    def __init__(self,size1,size2):
        super(SiameseNetwork,self).__init__()
        self.b1d32 = nn.BatchNorm1d(size2)
        self.bn1d = nn.BatchNorm1d(size1)

        self.c1 = nn.Linear(16384,size1)
        self.c2 = nn.Linear(size1,size2)
        self.c3 = nn.Linear(size2,1)


    def forward(self,input1,input2):
        x = torch.sub(input1,input2)
        x = torch.abs(x)
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
    o = 1
    if ids[i1,-1] == ids[i2,-1] and not ids[i1,-1] == "new_whale" and not ids[i1,-1] == "":
      o = 0
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

def cleanDataset(tr_ids):
    tr_idx= []
    for i in range(len(tr_ids)):
        if not (tr_ids[i,-1] == "new_whale") and not (tr_ids[i,-1] == ""):
            tr_idx.append(i)
    print(len(tr_idx))
    return tr_idx


def train(epochs,learning_rate,batch_size,set1,ids1,size1=128,size2=32):
    model = SiameseNetwork(size1,size2).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    no_improve = 0
    idx = cleanDataset(ids1)
    set1 = set1[idx]
    ids1 = ids1[idx]
    val_set_size = int(len(ids1)/3)
    val_set = set1[:val_set_size]
    val_ids = ids1[:val_set_size]
    set1 = set1[val_set_size:]
    ids1 = ids1[val_set_size:]
    model.train()
    mse = nn.MSELoss()
    es = EarlyStopper(patience=50,delta=0.001,save_path="siamese_network_ae_correct_v2.pth",save=True)
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
        if epoch%5 == 0:
          val_loss = 0
          for i in range(int(np.ceil(len(val_set)/batch_size))):
            b1,b2,l = createBatch(batch_size,val_set,val_ids)
            o = model(b1,b2)
            loss = mse(o,l)
            val_loss += loss
          stop = es.earlyStopping(val_loss,model)
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
    tr_enc = np.load("../ae/ae_training_encodings_simple.npy")
    tr_ids = np.load("../ae/ae_training_ids_simple.npy")
    tr_idx = findIndices(tr_ids)
    xs_tr_enc = tr_enc[tr_idx]
    xs_tr_ids = tr_ids[tr_idx]
    train(350,0.0001,64,xs_tr_enc,xs_tr_ids)


def matchTop10():
    model = torch.load("siamese_network_ae_correct_v2.pth").cuda()
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

    pred = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(te_ids)):
      if i%1000 == 0 and i > 0:
        if tp+fp > 0 and tp+fn > 0:
          print("accuracy: "+ str(np.mean(np.array(pred))) + "recall: " + str(tp/(tp+fn)) + " and precision: " + str(tp/(tp+fp)))
      te = torch.from_numpy(np.array([te_enc[i],te_enc[i],te_enc[i],te_enc[i],te_enc[i]])).float().cuda()
      matches = []
      for j in range(0,len(tr_ids)-5,5):
        tr = torch.from_numpy(np.array([tr_enc[j],tr_enc[j+1],tr_enc[j+2],tr_enc[j+3],tr_enc[j+4]])).float().cuda()
        res = model(te,tr)
        for r_idx in range(len(res)):
          r = res[r_idx]
          matches.append([r.item(),tr_ids[j+r_idx,-1]])
          if r < 0.5: #classified match 
            if te_ids[i,-1] == tr_ids[j+r_idx,-1]: #and is match
              tp += 1
            else: #but is no match
              fp += 1
          else: #classified not match
            if te_ids[i,-1] == tr_ids[j+r_idx,-1]: #but is match
              fn += 1
            else: #and is no match
              tn += 1
      matches = np.array(matches)
      if len(matches) > 0:
        matches = matches[matches[:,0].argsort()] #sorts from low numbers to high numbers 
      match = -1
      for k in range(0,10): #perfect, coz we want them low low numbers
        if k < len(matches):
          m_id = matches[k,-1]
          if te_ids[i,-1] == m_id:
            match = k
            break
        if match >= 0:
          pred.append(1)
        else: 
          pred.append(0)
      #if tp+fp > 0 and tp+fn > 0:
        #print("recall: " + str(tp/(tp+fn)) + " and precision: " + str(tp/(tp+fp)))
    a = np.mean(np.array(pred))
    print("test accuracy: " + str(a))
    print("TP: " + str(tp))
    print("TP?: " + str(np.sum(np.array(pred))))
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    print("recall: " + str(recall) + " and precision: " + str(precision))
    np.save("siamese_matches.npy",matches)

def objective(trial):
    epochs = 1000
    #learning_rate =trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    #batch_size = trial.suggest_int("batch_size",8,32,8)
    learning_rate = 0.0001
    batch_size = 64
    set1 = np.load("../ae/ae_training_encodings_simple.npy")
    ids1 = np.load("../ae/ae_training_ids_simple.npy")
    size1 = trial.suggest_categorical("size1",[1024,512,256])
    size2 = trial.suggest_categorical("size2",[32,64,128])
    #print("BATCH SIZE: " + str(batch_size))
    print(str(size1) + " " +str(size2))
    model = SiameseNetwork(size1,size2).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    no_improve = 0
    model.train()
    mse = nn.MSELoss()
    es = EarlyStopper(patience=50,delta=0.001,save_path="siamese_network_ae_correct.pth",save=False)
    for epoch in range(epochs):
        total_train_loss = 0
        optimizer.zero_grad()
        labels = []
        predictions = []
        for i in range(int(np.ceil(len(set1)/batch_size*2))):
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


def main():
    torch.cuda.set_device(0)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=5)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


torch.cuda.empty_cache()
#trial()
#train(500,0.0001,64,np.load("../ae/ae_training_encodings_simple.npy"),np.load("../ae/ae_training_ids_simple.npy"))
matchTop10()
#main()