import torch
from torch import nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score,recall_score
import optuna
from helper import cleanDataset,randomSubset

"""
Siamese Network with the given inputsize that takes two inputs, takes their element-wise squared differnce 
and feeds it through the network first downsizing to size1 then size2 before
returning a single distance value
"""
class SiameseNetwork(nn.Module):
    def __init__(self,inputsize,size1,size2):
        super(SiameseNetwork,self).__init__()
        self.b1d32 = nn.BatchNorm1d(size2)
        self.bn1d = nn.BatchNorm1d(size1)
        self.c1 = nn.Linear(inputsize,size1)
        self.c2 = nn.Linear(size1,size2)
        self.c3 = nn.Linear(size2,1)

    def forward(self,input1,input2):
        x = torch.sub(input1,input2)
        x = torch.square(x)
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

"""
Creates two batches of images and one batch of labels,
these labels state wether the individual in the two images 
is the same or not. Takes as input the batchsize, the encodings, ids
and everyX which determines how often the images in both image 
batches should be of the same individual
Returns three torch tensors two of images and one of labels 
"""
def createBatch(batch_size,enc,ids,everyX=2):
  def find_same(id_,idx):
    for i in range(len(ids)):
      if ids[i,-1] == id_ and not (idx == i):
        return i
    return -1

  def find_same_two():
    i1 = np.random.randint(0,len(ids))
    i2 = find_same(ids[i1,-1],i1)
    while i1 == i2 or i2 == -1:
      i1 = np.random.randint(0,len(ids))
      i2 = find_same(ids[i1,-1],i1)
    return i1,i2

  b1 = []
  b2 = []
  labels = []
  for i in range(batch_size):
    if i % everyX == 0:
      i1,i2 = find_same_two()
    else:
      i1 = np.random.randint(0,len(ids))
      i2 = np.random.randint(0,len(ids))
      while i1 == i2:
        i2 = np.random.randint(0,len(ids))
    o = 1
    if ids[i1,-1] == ids[i2,-1] and not ids[i1,-1] == "new_whale" and not ids[i1,-1] == "":
      o = 0
    b1.append(enc[i1])
    b2.append(enc[i2])
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


"""
Loads the siamese net at the net_path (.pth file) and calculates the top 10 accuracy
of the test set te_enc_path and te_ids_path (both .npy files) 
based on the training set tr_enc_path and tr_ids_path (both .npy files)
Returns (and prints) the top 10 accuracy of the test set
"""
def top10Siamese(net_path,tr_enc_path,tr_ids_path,te_enc_path,te_ids_path,reduced=None):
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
    pred = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    model = torch.load(net_path).cuda()
    model.eval()
    for i in range(len(te_ids)):
      te = torch.from_numpy(np.array([te_enc[i],te_enc[i]])).float().cuda()
      matches = []
      for j in range(0,len(tr_ids)):
        tr = torch.from_numpy(np.array([tr_enc[j],te_enc[i]])).float().cuda()
        r = model(te,tr)
        r = r[0]
        matches.append([r.item(),tr_ids[j,-1]])
        if r < 0.5: #classified match 
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
      if len(matches) > 0:
        matches = matches[matches[:,0].argsort()] #sorts from low numbers to high numbers 
      match = False
      whales = []
      counter = 0
      while len(whales) < 10:
        m_id = matches[counter,-1]
        if not (m_id in whales):
          whales.append(m_id)
          if te_ids[i,-1] == m_id: 
            match = True
            break
        counter += 1
      if match:
        pred.append(1)
      else: 
        pred.append(0)
    a = np.mean(np.array(pred))
    print("test accuracy: " + str(a))
    return a


"""
Creates a new Siamese Network with the hidden layer sizes size1 and size2 
the batch_size and learning_rate for epochs epochs
using the training data from the path tr_enc_path and tr_ids_path (both .npy files)
 of which the share of validation_split
is used as a validation set every 5 epochs
The network trained network is saved at the save_path
"""
def train(epochs,learning_rate,batch_size,tr_enc_path,tr_ids_path,save_path="siamese_network_vae_correct_v2_2.pth",size1=128,size2=32,validation_split=1/3):
    set1 = np.load(tr_enc_path)
    ids1 = np.load(tr_ids_path)
    tr_idx = cleanDataset(ids1)
    set1 = set1[tr_idx]
    ids1 = ids1[tr_idx]
    validation_split = 1/3
    val_set_size = int(len(ids1)*validation_split)
    val_set = set1[:val_set_size]
    val_ids = ids1[:val_set_size]
    set1 = set1[val_set_size:]
    ids1 = ids1[val_set_size:]
    inputsize = len(set1[0])
    model = SiameseNetwork(inputsize,size1,size2).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    bce = nn.BCELoss()
    es = EarlyStopper(patience=20,delta=0.1,save_path=save_path,save=True)
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []
    for epoch in range(epochs):
        total_train_loss = 0
        optimizer.zero_grad()
        labels = []
        predictions = []
        model.train()
        for i in range(int(np.ceil(len(set1)/batch_size))):
            b1,b2,l = createBatch(batch_size,set1,ids1)
            o = model(b1,b2)
            loss = bce(o,l)
            loss.backward()
            optimizer.step()
            total_train_loss += loss
            labels.extend(l[:,0].tolist())
            predictions.extend(o[:,0].tolist())
        training_losses.append(loss.item())
        a = accuracy_score(labels,np.where(np.array(predictions) < 0.5, 0.0,1.0))
        training_accuracies.append(a)
        r = recall_score(labels,np.where(np.array(predictions) < 0.5, 0.0,1.0))
        stop_epoch = epoch
        if epoch%5 == 0:
            val_loss = 0
            vpredictions = []
            model.eval()
            vlabels = []
            with torch.no_grad():
                for i in range(int(np.ceil(len(val_set)/batch_size))):
                    b1,b2,l = createBatch(batch_size,val_set,val_ids)
                    o = model(b1,b2)
                    loss = bce(o,l)
                    val_loss += loss
                    vlabels.extend(l[:,0].tolist())
                    vpredictions.extend(o[:,0].tolist())
                va = accuracy_score(vlabels,np.where(np.array(vpredictions) < 0.5, 0.0,1.0))
                vr = recall_score(vlabels,np.where(np.array(vpredictions) < 0.5, 0.0,1.0))
                print("EPOCH " + str(epoch) + " with loss "  + str(val_loss.item()) + ", accuracy " + str(va) + " and recall " + str(vr))
                stop = es.earlyStopping(val_loss,model)
                validation_losses.append(val_loss.item())
                validation_accuracies.append(va)
                if stop:
                    print("TRAINING FINISHED AFTER " + str(epoch) + " EPOCHS. K BYE.")
                    break
    if (int(stop_epoch/5)-20) < len(validation_losses):
        final_loss = validation_losses[int(stop_epoch/5)-20] #every 5 epochs validation and 20 coz of patience
        final_accuracy = validation_accuracies[int(stop_epoch/5)-20]
    else:
        final_loss = validation_losses[-1]
        final_accuracy = validation_accuracies[-1]
    #WRITE OPTIM 
    filename = str("siamese_optim_losses_v2.txt")
    file=open(filename,'a')
    file.write("Training loss:")
    file.write('\n')
    for l in training_losses:
        file.write(str(l))
        file.write('\n')
    file.write("Training accuracy:")
    file.write('\n')
    for l in training_accuracies:
        file.write(str(l))
        file.write('\n')
    file.write("Validation loss:")
    file.write('\n')
    for l in validation_losses:
        file.write(str(l))
        file.write('\n')
    file.write("Validation accuracy:")
    file.write('\n')
    for l in validation_accuracies:
        file.write(str(l))
        file.write('\n')
    file.write("final_loss:" + str(final_loss)) 
    file.write("final_accuracy:" + str(final_accuracy))
    file.write('\n') 
    file.close()


def objective(trial):
    epochs = 1000
    #learning_rate =trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    #batch_size = trial.suggest_int("batch_size",8,32,8)
    learning_rate = 0.00001
    batch_size = 64
    size1 = 512#trial.suggest_categorical("size1",[1024,512,256])
    size2 = 64#trial.suggest_categorical("size2",[32,64,128])
    print("Size1:" + str(size1))
    print("Size2:" + str(size2))
    set1 = np.load("../ae/ae_training_encodings_simple_v2.npy")
    ids1 = np.load("../ae/ae_training_ids_simple_v2.npy")
    tr_idx = cleanDataset(ids1)
    set1 = set1[tr_idx]
    ids1 = ids1[tr_idx]
    validation_split = 1/3
    val_set_size = int(len(ids1)*validation_split)
    val_set = set1[:val_set_size]
    val_ids = ids1[:val_set_size]
    set1 = set1[val_set_size:]
    ids1 = ids1[val_set_size:]
    inputsize = len(set1[0])
    model = SiameseNetwork(inputsize,size1,size2).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    bce = nn.BCELoss()
    es = EarlyStopper(patience=20,delta=0.1,save_path="siamese.pth",save=False)
    validation_losses = []
    validation_accuracies = []
    for epoch in range(epochs):
        total_train_loss = 0
        optimizer.zero_grad()
        labels = []
        predictions = []
        model.train()
        for i in range(int(np.ceil(len(set1)/batch_size))):
            b1,b2,l = createBatch(batch_size,set1,ids1)
            o = model(b1,b2)
            loss = bce(o,l)
            loss.backward()
            optimizer.step()
            total_train_loss += loss
            labels.extend(l[:,0].tolist())
            predictions.extend(o[:,0].tolist())
        a = accuracy_score(labels,np.where(np.array(predictions) < 0.5, 0.0,1.0))
        r = recall_score(labels,np.where(np.array(predictions) < 0.5, 0.0,1.0))
        stop_epoch = epoch
        if epoch%5 == 0:
            val_loss = 0
            vpredictions = []
            model.eval()
            vlabels = []
            with torch.no_grad():
                for i in range(int(np.ceil(len(val_set)/batch_size))):
                    b1,b2,l = createBatch(batch_size,val_set,val_ids)
                    o = model(b1,b2)
                    loss = bce(o,l)
                    val_loss += loss
                    vlabels.extend(l[:,0].tolist())
                    vpredictions.extend(o[:,0].tolist())
                va = accuracy_score(vlabels,np.where(np.array(vpredictions) < 0.5, 0.0,1.0))
                vr = recall_score(vlabels,np.where(np.array(vpredictions) < 0.5, 0.0,1.0))
                print("EPOCH " + str(epoch) + " with loss "  + str(val_loss.item()) + ", accuracy " + str(va) + " and recall " + str(vr))
                stop = es.earlyStopping(val_loss,model)
                trial.report(val_loss,epoch)
                validation_losses.append(val_loss.item())
                validation_accuracies.append(va)
                if stop:
                    print("TRAINING FINISHED AFTER " + str(epoch) + " EPOCHS. K BYE.")
                    break
    if (int(stop_epoch/5)-20) < len(validation_losses):
        final_loss = validation_losses[int(stop_epoch/5)-20] #every 5 epochs validation and 20 coz of patience
        final_accuracy = validation_accuracies[int(stop_epoch/5)-20]
    else:
        final_loss = validation_losses[-1]
        final_accuracy = validation_accuracies[-1]
    #WRITE OPTIM 
    filename = str("siamese_optim_v2.txt")
    file=open(filename,'a')
    file.write("size1:" + str(size1))
    file.write("size2:" + str(size2))
    file.write("final_loss:" + str(final_loss)) 
    file.write("final_accuracy:" + str(final_accuracy))
    file.write('\n') 
    file.close()
    return final_loss


def optimal_optimisation():
    torch.cuda.set_device(0)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=1)

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

