import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.datasets as datasets

import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import random
from skimage import io, util, transform, color,exposure,filters
from sklearn.metrics import accuracy_score
import optuna

from dataset import getDatasets, WhaleDataset
from earlystopper import EarlyStopper

torch.set_default_tensor_type('torch.cuda.FloatTensor')

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
"""
The facenet Siamese Autoencoder module with the following parameters:
layer_amount = the amount of regular layers in the encoder, possible values 4,5,6, default 6
channels = the amount of channels of the input image, default 1
isize = the size of the square input image, default 512 for 512x512
layer_size = the amount of feature maps at the bottleneck, possible vlaues 32,64,128,256, default 128
size1 = the size of the first fully connected layer of the Siamese Network, default 256
size2 = the size of the second fully connected layer of the Siamese Network, default 32
"""
class facenetAE(nn.Module):
    def __init__(self,layer_amount=5,channels=1,isize=512,layer_size=64,extradense=False,size1=256,size2=32):
        super(facenetAE,self).__init__()
        poolamount = 16
        h_dim = int(pow((isize/(poolamount*2)),2)*layer_size)
        self.layer_amount = layer_amount
        self.layer_size=layer_size
        self.extradense = extradense

        #Encoder
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3)  
        self.conv2 = nn.Conv2d(64, 64, 1,stride=1)
        self.conv2a = nn.Conv2d(64, 192,3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(192, 192,1,stride=1)
        self.conv3a = nn.Conv2d(192,384,3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(384,384,1,stride=1)
        self.conv4a = nn.Conv2d(384,256,3,stride=1,padding=1)
        if(self.layer_amount > 4):
            self.conv5 = nn.Conv2d(256,256,1,stride=1)
            self.conv5a = nn.Conv2d(256,256,3,stride=1,padding=1)
            torch.nn.init.xavier_uniform_(self.conv5.weight,gain=nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_uniform_(self.conv5a.weight,gain=nn.init.calculate_gain('relu'))
            if(self.layer_amount > 5):
                self.conv6 = nn.Conv2d(256,256,1,stride=1)
                self.conv6a = nn.Conv2d(256,256,3,stride=1,padding=1)
                torch.nn.init.xavier_uniform_(self.conv6.weight,gain=nn.init.calculate_gain('relu'))
                torch.nn.init.xavier_uniform_(self.conv6a.weight,gain=nn.init.calculate_gain('relu'))

        if(self.layer_size == 128):
            self.conv41 = nn.Conv2d(256,128,1,stride=1)
            self.conv41a = nn.Conv2d(128,128,3,stride=1,padding=1)
            self.t_conv41 = nn.ConvTranspose2d(128,256,1,stride=1)
            torch.nn.init.xavier_uniform_(self.conv41.weight,gain=nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_uniform_(self.conv41a.weight,gain=nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_uniform_(self.t_conv41.weight,gain=nn.init.calculate_gain('relu'))
        elif(self.layer_size == 64):
            self.conv42 = nn.Conv2d(256,64,1,stride=1)
            self.conv42a = nn.Conv2d(64,64,3,stride=1,padding=1)
            self.t_conv42 = nn.ConvTranspose2d(64,256,1,stride=1)
            torch.nn.init.xavier_uniform_(self.conv42.weight,gain=nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_uniform_(self.conv42a.weight,gain=nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_uniform_(self.t_conv42.weight,gain=nn.init.calculate_gain('relu'))
        elif(self.layer_size == 32):
            self.conv43 = nn.Conv2d(256,32,1,stride=1)
            self.conv43a = nn.Conv2d(32,32,3,stride=1,padding=1)
            self.t_conv43 = nn.ConvTranspose2d(32,256,1,stride=1)
            torch.nn.init.xavier_uniform_(self.conv43.weight,gain=nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_uniform_(self.conv43a.weight,gain=nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_uniform_(self.t_conv43.weight,gain=nn.init.calculate_gain('relu'))

        self.pool = nn.MaxPool2d(3, 2,padding=1)
        self.lrn = nn.LocalResponseNorm(2)
       
        #Bottleneck
        self.fl = Flatten()
        if self.extradense:
            self.extra = nn.Linear(h_dim,h_dim)
        self.unfl = UnFlatten(layer_size)

        #SIAMESE
        self.b1d32 = nn.BatchNorm1d(size2)
        self.bn1d = nn.BatchNorm1d(size1)
        self.c1 = nn.Linear(h_dim,size1)
        self.c2 = nn.Linear(size1,size2)
        self.c3 = nn.Linear(size2,1)

        #Decoder        
        self.t_conv4 = nn.ConvTranspose2d(256,384,2,stride=2)
        self.t_conv3 = nn.ConvTranspose2d(384,192,2,stride=2)
        self.t_conv2 = nn.ConvTranspose2d(192,64,2,stride=2)
        self.t_conv1 = nn.ConvTranspose2d(64, 1, 4, stride=4)

        #Initialise Weights
        torch.nn.init.xavier_uniform_(self.conv1.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv2.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv2a.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv3.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv3a.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv4.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv4a.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.t_conv4.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.t_conv3.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.t_conv2.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.t_conv1.weight,gain=nn.init.calculate_gain('relu'))

    """
    Encodes an image x using the encoder.
    Returns the encoding.
    """
    def encode(self,x):
        x = nn.ReLU()(self.conv1(x))
        x = self.pool(x)
        x = self.lrn(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv2a(x))
        x = self.lrn(x)
        x = self.pool(x)
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv3a(x))
        x = self.pool(x)
        x = nn.ReLU()(self.conv4(x))
        x = nn.ReLU()(self.conv4a(x))
        if(self.layer_amount > 4):
            x = nn.ReLU()(self.conv5(x))
            x = nn.ReLU()(self.conv5a(x))
            if(self.layer_amount > 5):
                    x = nn.ReLU()(self.conv6(x))
                    x = nn.ReLU()(self.conv6a(x))
        if(self.layer_size == 128):
            x = nn.ReLU()(self.conv41(x))
            x = nn.ReLU()(self.conv41a(x))
        elif(self.layer_size == 64):
            x = nn.ReLU()(self.conv42(x))
            x = nn.ReLU()(self.conv42a(x))
        elif(self.layer_size == 32):
            x = nn.ReLU()(self.conv43(x))
            x = nn.ReLU()(self.conv43a(x))
        x = self.pool(x)
        h=self.fl(x)
        if self.extradense:
            h = self.extra(h)
        return(h)

    """
    Throughputs two inputs (both are encodings) through the siamese network.
    Returns the distance value between these two inputs.
    """
    def siamese(self,input1,input2,batched=True):
        x = torch.sub(input1,input2)
        x = torch.square(x)
        x = nn.ReLU()(self.c1(x))
        if batched:
            x = self.bn1d(x)
            x = nn.Dropout2d(0.25)(x)
        x = nn.ReLU()(self.c2(x))
        if batched:
            x = self.b1d32(x)
        x = nn.Sigmoid()(self.c3(x))
        return(x)

    """
    Forwards an image through the encoder and decoder. 
    Returns the reconstructed image.
    """
    def forward(self, x):
        h = self.encode(x)
        x=self.unfl(h)
        if(self.layer_size == 128):
            x = nn.ReLU()(self.t_conv41(x))
        elif(self.layer_size == 64):
            x = nn.ReLU()(self.t_conv42(x))
        elif(self.layer_size == 32):
            x = nn.ReLU()(self.t_conv43(x))
        x = nn.ReLU()(self.t_conv4(x))
        x = nn.ReLU()(self.t_conv3(x))
        x = nn.ReLU()(self.t_conv2(x))
        x = self.t_conv1(x)
        x = nn.Sigmoid()(x)
        return x

    """
    Forwards two images through the encoder and the siamese network.
    Returns the distance between both images.
    """
    def forward_siamese(self,x1,x2):
        h1 = self.encode(x1)
        h2 = self.encode(x2)
        x = self.siamese(h1,h2)
        return x

    """
    Forwards image x1 through the encoder and decoder.
    Returns the reconstructed image of x1.
    Encodes x2 and runs the encodings of x1 and x2 through the siamese network.
    Returns the distance of both images.
    Output looks as follows reconstructed image x1, distance of x1 and x2.
    """
    def full_forward(self,x1,x2):
        h = self.encode(x1)
        x=self.unfl(h)
        if(self.layer_size == 128):
            x = nn.ReLU()(self.t_conv41(x))
        elif(self.layer_size == 64):
            x = nn.ReLU()(self.t_conv42(x))
        elif(self.layer_size == 32):
            x = nn.ReLU()(self.t_conv43(x))
        x = nn.ReLU()(self.t_conv4(x))
        x = nn.ReLU()(self.t_conv3(x))
        x = nn.ReLU()(self.t_conv2(x))
        x = self.t_conv1(x)
        x = nn.Sigmoid()(x)
        h2 = self.encode(x2)
        d = self.siamese(h,h2)
        return x,d

"""
Calculates the MSE loss between the reconstructed image recon_x and the original input x.
Returns the loss.
"""
def loss_fn(recon_x, x):
    loss = nn.MSELoss()
    l = loss(recon_x,x)
    return l

"""
Calculates the BCE loss between the reconstructed image output array and the target array.
Returns the loss.
"""
def loss_siam(output,target):
    loss = nn.BCELoss()
    l = loss(output,target)
    return l

"""
Creates and trains the facenetAE network given the parameters:
epochs = amount of maximum epochs
learning_rate = the learning rate used to train the network
batch_size = the batch size used to train the network
data_path = the path to the training data set
layers = the amount of regular layers (possible values = 4,5,6)
layer_size = the amount of feature maps at the bottleneck (possible values = 32,64,128,256)
factor = the factor that weighs the loss of the siamese network against that of the autoencoder, default 6
save = a boolean whether the network should be saved or not, default True
trainFrom = whether the network should be trained from a safepoint, default False
"""
def trainNet(epochs,learning_rate,batch_size,data_path,layers,layer_size,factor=6,save=True,trainFrom=False):
    train_loader,val_loader,train_set,val_set = getDatasets(data_path,batch_size,raw=True,findDoubles=True)
    savepath= str("AE_earlystopsave_simple_siamese_v3.pth")
    #TRAINING
    if trainFrom:
        model = torch.load(savepath)
        trainFromEpoch=np.load("ae_current_epoch.npy")[0]
        training_losses = np.load("ae_earlystopsave_training_losses.npy")
        siamese_losses = np.load("ae_earlystopsave_siamese_losses.npy")
        validation_losses = np.load("ae_earlystopsave_validation_losses.npy")
        validation_accs = np.load("ae_earlystopsave_validation_accs.npy")
    else:
        model = facenetAE(layer_amount=layers,layer_size=layer_size).cuda()
        training_losses = np.array([])
        siamese_losses = np.array([])
        validation_losses = np.array([])
        validation_accs = np.array([])
        trainFromEpoch = 0
    es = EarlyStopper(10,1,savepath,save)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(trainFromEpoch,epochs):
        total_train_loss = 0
        model.train()
        siamese_loss = 0
        if epoch % 2 == 0: #first half
            datasethalf = range(0,int(math.ceil(train_set.getDatasetSize()/batch_size)/2))
        else: #second half
            datasethalf = range(int(math.ceil(train_set.getDatasetSize()/batch_size)/2),int(math.ceil(train_set.getDatasetSize()/batch_size)))
        for b in datasethalf:
            optimizer.zero_grad()
            b1,b2,l = train_set.getDoubleBatch(batch_size,b)
            o = model.forward_siamese(b1,b2)
            loss_s = loss_siam(o,l)
            o1 = model.forward(b1)
            loss_t = loss_fn(o1,b1)
            loss = loss_s * factor + loss_t
            loss.backward()
            optimizer.step()
            siamese_loss += loss_s
            total_train_loss += loss
        siamese_losses = np.append(siamese_losses,siamese_loss.detach().cpu().item())
        training_losses = np.append(training_losses,total_train_loss.detach().cpu().item()) 
        print("siamese loss "  + str(siamese_loss.detach().cpu().item()))
        print("train loss " + str(total_train_loss.detach().cpu().item()))
        stop_epoch = epoch
        #VALIDATION 
        if epoch % 10 == 0: 
            model.eval()
            vpredictions = []
            vlabels = []
            with torch.no_grad():
                total_val_loss = 0
                for b in range(int(math.ceil(train_set.getDatasetSize()/batch_size))):
                    b1,b2,l = train_set.getDoubleBatch(batch_size,b)
                    #o = model.forward_siamese(b1,b2)
                    o1,d = model.full_forward(b1,b2)
                    loss_s = loss_siam(d,l)
                    loss_t = loss_fn(o1,b1)
                    loss = loss_s * factor + loss_t
                    siamese_loss += loss_s
                    total_val_loss += loss
                    vlabels.extend(l[:,0].tolist())
                    vpredictions.extend(o[:,0].tolist())
                va = accuracy_score(vlabels,np.where(np.array(vpredictions) < 0.5, 0.0,1.0))
                validation_accs.append(va)
            validation_losses = np.append(validation_losses,total_val_loss.detach().cpu().item())
            print("Epoch " + str(epoch) + " with val loss " + str(validation_losses[-1]))
            stop = es.earlyStopping(total_val_loss,model)
            #EARLY STOPPING
            if stop:
                print("TRAINING FINISHED AFTER " + str(epoch) + " EPOCHS. K BYE.")
                break
            else:
                if save:
                    np.save("ae_earlystopsave_training_losses.npy",training_losses)
                    np.save("ae_earlystopsave_siamese_losses.npy",siamese_losses)
                    np.save("ae_earlystopsave_validation_losses.npy",validation_losses)
                    np.save("ae_earlystopsave_validation_accs.npy",validation_accs)
                    np.save("ae_current_epoch.npy",np.array([epoch]))
    #SAVE LOSSES TO FILE
    if save:
        filename = str("AE_losses_siamese_simple_v3.txt")
        file=open(filename,'w')
        file.write("trained with learning rate " + str(learning_rate) + ", batch size " + str(batch_size) + ", planned epochs " + str(epochs) + " but only took " + str(stop_epoch) + " epochs.")
        file.write("training_losses")
        file.write('\n')
        for element in training_losses:
            file.write(str(element))
            file.write('\n')
        file.write("siamese_losses")
        file.write('\n')
        for element in siamese_losses:
            file.write(str(element))
            file.write('\n')
        file.write("validation_losses")
        file.write('\n')
        for element in validation_losses:
            file.write(str(element))
            file.write('\n')   
        file.write("validation_accuracies")
        for element in validation_accs:
            file.write(str(element))
            file.write('\n')   
        file.close()

"""
Evaluates a network at the network_path (default none), given the test set at filepath.
Prints the total loss of the test set in the console.
"""
def evalSet(filepath,network_path=None):
    #get data
    loader, v_loader = getDatasets(filepath,4,validation_split=0,reduction=1,raw=False,augment=False)
    #get model
    if network_path:
        model = torch.load(network_path)
    else:
        model = facenetAE()
    #calculate loss
    total_loss = 0
    model.eval()
    for inputs in loader:
        inputs = inputs.float().cuda()
        outputs = model.forward(inputs)
        lost  = loss_fn(outputs,inputs)
        total_loss += lost.detach().cpu().item()
    print("Total loss for testset is: " + str(total_loss))

"""
Loads the siamese net at the net_path (.pth file) and calculates the top 10 accuracy
of the already processed encodings of the test and training set,
(input as the tr_ and te_ path which are the paths to the encodings
and the tr_ids_ and te_ids_path which are the paths to the id files, all .npy files)
Returns (and prints) the top 10 accuracy of the test set
"""
def top10Siamese(net_path,tr_path,tr_ids_path,te_path,te_ids_path):
    train_enc = np.load(tr_path)
    train_ids = np.load(tr_ids_path)
    test_enc = np.load(te_path)
    test_ids = np.load(te_ids_path)
    te_len = len(test_ids)
    tr_len = len(train_ids)
    pred = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    model = torch.load(net_path).cuda()
    model.eval()
    for i in range(te_len):
      te = test_enc[i]
      _,_,te_id = test_ids[i]
      matches = []
      te = torch.from_numpy(te).float().cuda()
      for j in range(tr_len):
        tr = train_enc[j]
        _,_,tr_id = train_ids[j]
        tr = torch.from_numpy(tr).float().cuda()
        r = model.siamese(te,tr,batched=False)
        r = r[0]
        matches.append([r.item(),tr_id])
        if r < 0.5: #classified match 
            if te_id == tr_id: #and is match
              tp += 1
            else: #but is no match
              fp += 1
        else: #classified not match
            if te_id == tr_id: #but is match
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
          if te_id == m_id: 
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
The objective for the optimisation.
"""
def objective(trial):
    epochs = 300
    data_path="../data/trainingset_final_v2.csv"
    learning_rate = 0.0001 #trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = 4 #trial.suggest_int("batch_size",8,32,8)
    layer_amount = trial.suggest_int("layer_amount",4,6,1)
    layer_size = trial.suggest_categorical("layer_size",[32,64,128])#,256])
    extradense = False #trial.suggest_categorical("extradense",[True,False])
    factor = trial.suggest_int("factor",1,30)
    print("BATCH SIZE: " + str(batch_size))
    print("learning rate: " + str(learning_rate))
    print("layer amount: " + str(layer_amount))
    print("layer size: " + str(layer_size))
    print("extradense: " + str(extradense))
    print("Factor " + str(factor))
    train_loader,val_loader,train_set,val_set = getDatasets(data_path,batch_size,reduction=0.25,raw=True,findDoubles=True)
    model = facenetAE(layer_amount=layer_amount,layer_size=layer_size,extradense=extradense).cuda()
    es = EarlyStopper(10,0.1,str("AE_earlystopsave_siamese_simple_v2.pth"),False)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #TRAINING
    training_losses = []
    siamese_losses = []
    validation_losses = []
    validation_accs = []
    for epoch in range(epochs):
        total_train_loss = 0
        model.train()
        siamese_loss = 0
        for b in range(int(math.ceil(train_set.getDatasetSize()/batch_size))):
            optimizer.zero_grad()
            b1,b2,l = train_set.getDoubleBatch(batch_size,b)
            o = model.forward_siamese(b1,b2)
            loss_s = loss_siam(o,l)
            o1 = model.forward(b1)
            loss_t = loss_fn(o1,b1)
            loss = loss_s * factor + loss_t
            loss.backward()
            optimizer.step()
            siamese_loss += loss_s
            total_train_loss += loss_t
        siamese_losses.append(siamese_loss.detach().cpu().item())
        training_losses.append(total_train_loss.detach().cpu().item()) 
        print("siamese loss "  + str(siamese_loss.detach().cpu().item()))
        print("train loss " + str(total_train_loss.detach().cpu().item()))
        stop_epoch = epoch
        #VALIDATION 
        if epoch % 10 == 0: 
            model.eval()
            vpredictions = []
            vlabels = []
            with torch.no_grad():
                total_val_loss = 0
                for b in range(int(math.ceil(train_set.getDatasetSize()/batch_size))):
                    b1,b2,l = train_set.getDoubleBatch(batch_size,b)
                    o = model.forward_siamese(b1,b2)
                    loss_s = loss_siam(o,l)
                    o1 = model.forward(b1)
                    loss_t = loss_fn(o1,b1)
                    loss = loss_s * factor + loss_t
                    siamese_loss += loss_s
                    total_val_loss += loss
                    vlabels.extend(l[:,0].tolist())
                    vpredictions.extend(o[:,0].tolist())
                va = accuracy_score(vlabels,np.where(np.array(vpredictions) < 0.5, 0.0,1.0))
                validation_accs.append(va)
            validation_losses.append(total_val_loss.detach().cpu().item())
            print("Epoch " + str(epoch) + " with val loss " + str(validation_losses[-1]) + " and val accuracy " + str(validation_accs[-1]))
            trial.report(total_val_loss,epoch)
            stop = es.earlyStopping(total_val_loss,model)
            #EARLY STOPPING
            if stop:
                print("TRAINING FINISHED AFTER " + str(epoch) + " EPOCHS. K BYE.")
                break
    if (int(stop_epoch/10)-10) < len(validation_losses):
        final_loss = validation_losses[int(stop_epoch/10)-10] #10 coz of patience = 10
        final_acc = validation_accs[int(stop_epoch/10)-10]
    else:
        final_loss = validation_losses[-1]
    #WRITE OPTIM 
    filename = str("ae_optim_v2.txt")
    file=open(filename,'a')
    file.write("factor:" + str(factor))
    file.write(" final_acc:" + str(final_acc))
    file.write(" final_loss:" + str(final_loss)) 
    file.write('\n') 
    file.close()
    return final_loss

"""
The optimisation of the network architecture. Runs for x trials trying to find the validation loss.
"""
def optimal_optimisation(x):
    torch.cuda.set_device(0)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=x)

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