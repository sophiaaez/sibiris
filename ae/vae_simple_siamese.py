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
from datetime import datetime

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
The facenet SiameseVariational Autoencoder module with the following parameters:
layer_amount = the amount of regular layers in the encoder, possible values 4,5,6, default 6
channels = the amount of channels of the input image, default 1
isize = the size of the square input image, default 512 for 512x512
layer_size = the amount of feature maps at the bottleneck, possible vlaues 32,64,128,256, default 32
size1 = the size of the first fully connected layer of the Siamese Network, default 256
size2 = the size of the second fully connected layer of the Siamese Network, default 32
"""
class facenetVAE(nn.Module):
    def __init__(self,layer_amount=6,channels=1,isize=512,layer_size=32,size1=256,size2=32):
        super(facenetVAE,self).__init__()
        poolamount = 16
        h_dim = int(pow((isize/(poolamount*2)),2)*layer_size)
        z_dim = int(h_dim/4)
        self.layer_amount = layer_amount
        self.layer_size=layer_size
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
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.unfl = UnFlatten(layer_size)

        #SIAMESE
        self.b1d32 = nn.BatchNorm1d(size2)
        self.bn1d = nn.BatchNorm1d(size1)
        self.c1 = nn.Linear(z_dim,size1)
        self.c2 = nn.Linear(size1,size2)
        self.c3 = nn.Linear(size2,1)

        #Decoder
        self.t_conv41 = nn.ConvTranspose2d(128,256,1,stride=1)
        self.t_conv42 = nn.ConvTranspose2d(64,256,1,stride=1)
        self.t_conv43 = nn.ConvTranspose2d(32,256,1,stride=1)
        self.t_conv4 = nn.ConvTranspose2d(256,384,2,stride=2)
        self.t_conv3 = nn.ConvTranspose2d(384,192,2,stride=2)
        self.t_conv2 = nn.ConvTranspose2d(192,64,2,stride=2)
        self.t_conv1 = nn.ConvTranspose2d(64, 1, 4, stride=4)

        #weight initialisation
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
    Applying the reparametrisation trick and resampling a vector z from the mu and logvar
    Returns vector z.
    """        
    def reparameterize(self, mu, logvar):  # producing latent layer (Guassian distribution )
        std = torch.exp(0.5 * logvar) #logvar.mul(0.5).exp_()       # hint: var=std^2
        esp = torch.randn_like(std)#torch.randn(*mu.size()).cuda()   # normal unit distribution in shape of mu
        z = mu + torch.sqrt(std) * esp     # mu:mean  std: standard deviation
        return z

    """
    Calculates the mu and logvar for the latent vector h and resamples it from these vectors.
    Returns resampled vector z, mu and logvar.
    """
    def bottleneck(self, h):      # hidden layer ---> mean layer + logvar layer
        mu = self.fc1(h)
        logvar = self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    """
    Encodes an image x using the encoder.
    Returns the sampled vector z, the vector mu and the vector logvar.
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
        z,mu,logvar=self.bottleneck(h)
        return z,mu,logvar

    """
    Forwards an image through the encoder and decoder. 
    Returns the reconstructed image, the mu vector and the logvar vector.
    """
    def forward(self, x):
        z,mu,logvar = self.encode(x)        
        hz=self.fc3(z)
        x=self.unfl(hz)
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
        #print(x.size())
        return x,mu,logvar

    """
    Throughputs two inputs (both are encodings) through the siamese network.
    Returns the distance value between these two inputs.
    """
    def siamese(self,input1,input2):
        x = torch.sub(input1,input2)
        x = torch.square(x)
        x = nn.ReLU()(self.c1(x))
        x = self.bn1d(x)
        x = nn.Dropout2d(0.25)(x)
        x = nn.ReLU()(self.c2(x))
        x = self.b1d32(x)
        x = nn.Sigmoid()(self.c3(x))
        return(x)

    """
    Forwards two images through the encoder and uses their mean vectors 
    to run them through the siamese network.
    Returns the distance between both images.
    """
    def forward_siamese(self,x1,x2):
        _,h1,_ = self.encode(x1)
        _,h2,_ = self.encode(x2)
        x = self.siamese(h1,h2)
        return x

    """
    Forwards image x1 through the encoder and decoder.
    Returns the reconstructed image of x1, its mu vector and logvar vector.
    Uses the mu latent encoding of x2 and 
    runs the mu encodings of x1 and x2 through the siamese network.
    Returns the distance of both images.
    Output looks as follows reconstructed image x1, mu vector of x1, logvar vector of x1, 
    distance of x1 and x2.
    """
    def full_forward(self,x1,x2):
        z,mu,logvar = self.encode(x1)        
        hz=self.fc3(z)
        x=self.unfl(hz)
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
        _,mu2,_ = self.encode(x2)
        d = self.siamese(mu,mu2)
        return x,mu,logvar,d

"""
Calculates different losses between the reconstructed image recon_x and the original input x.
One loss is the mse (using the sum reduction), one is the kullback-leibler-divergence and one is the 
combination of both, weighted with the factor beta.
Returns the combined loss, the mse loss and the kld loss.
"""
def loss_fn(recon_x, x,mu,logvar,beta):   # defining loss function for va-AE (loss= reconstruction loss + KLD (to analyse if we have normal distributon))
    loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
    return (loss + beta * kld) / (x.shape[-1] * x.shape[-2]), loss, kld


def loss_siam(output,target):
    loss = nn.BCELoss()
    l = loss(output,target)
    return l

"""
Creates and trains the facenetVAE network given the parameters:
epochs = amount of maximum epochs
learning_rate = the learning rate used to train the network
batch_size = the batch size used to train the network
data_path = the path to the training data set
layers = the amount of regular layers (possible values = 4,5,6)
layer_size = the amount of feature maps at the bottleneck (possible values = 32,64,128,256)
factor = the factor that weighs the loss of the siamese network against that of the autoencoder, default 15
beta = the factor that weighs the mse and kld loss of the vae against each other, default 1
save = a boolean whether the network should be saved or not, default True
trainFrom = whether the network should be trained from a safepoint, default False
"""
def trainNet(epochs,learning_rate,batch_size,data_path,layers,layer_size,factor=15,beta=1,save=True,trainFrom=False):
    train_loader,val_loader,train_set,val_set = getDatasets(data_path,batch_size,raw=True,findDoubles=True)
    savepath= str("VAE_earlystopsave_simple_siamese_v3.pth")
    if trainFrom:
        trainFromEpoch = np.load("vae_current_epoch.npy")[0]
        model = torch.load(savepath)
        training_losses = np.load("vae_earlystopsave_training_losses.npy")
        mse_losses = np.load("vae_earlystopsave_mse_losses.npy")
        kld_losses = np.load("vae_earlystopsave_kld_losses.npy")
        siamese_losses = np.load("vae_earlystopsave_siamese_losses.npy")
        validation_losses = np.load("vae_earlystopsave_validation_losses.npy")
        validation_accs = np.load("vae_earlystopsave_validation_accs.npy")       
    else:
        model = facenetVAE(layer_amount=layers,layer_size=layer_size).cuda()
        training_losses = np.array([])
        mse_losses = np.array([])
        kld_losses = np.array([])
        siamese_losses = np.array([])
        validation_losses = np.array([])
        validation_accs = np.array([])
        trainFromEpoch = 0
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    es = EarlyStopper(10,1,savepath,save)
    #Training
    for epoch in range(trainFromEpoch,epochs):
        total_train_loss = 0
        model.train()
        siamese_loss = 0
        total_mse = 0
        total_kld = 0
        if epoch % 2 == 0: #first half
            datasethalf = range(0,int(math.ceil(train_set.getDatasetSize()/batch_size)/2))
        else: #second half
            datasethalf = range(int(math.ceil(train_set.getDatasetSize()/batch_size)/2),int(math.ceil(train_set.getDatasetSize()/batch_size)))
        for b in datasethalf:
            optimizer.zero_grad()
            b1,b2,l = train_set.getDoubleBatch(batch_size,b)
            #o = model.forward_siamese(b1,b2)
            o1,mu1,log1,d = model.full_forward(b1,b2)
            loss_s = loss_siam(d,l)
            loss_t,mse,kld = loss_fn(o1,b1,mu1,log1,beta)
            total_mse += mse
            total_kld += kld
            loss = loss_s * factor + loss_t
            loss.backward()
            optimizer.step()
            siamese_loss += loss_s
            total_train_loss += loss
        siamese_losses = np.append(siamese_losses,siamese_loss.detach().cpu().item())
        training_losses = np.append(training_losses,total_train_loss.detach().cpu().item()) 
        mse_losses = np.append(mse_losses,total_mse.detach().cpu().item())
        kld_losses = np.append(kld_losses,total_kld.detach().cpu().item())
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
                for b in range(int(math.ceil(val_set.getDatasetSize()/batch_size))):
                    b1,b2,l = val_set.getDoubleBatch(batch_size,b)
                    vlabels.extend(l[:,0].tolist())
                    o1,vmu,vlog,d = model.full_forward(b1,b2)
                    vpredictions.extend(d[:,0].tolist())
                    loss_s = loss_siam(d,l)
                    loss_t,_,_ = loss_fn(o1,b1,vmu,vlog,beta)
                    loss = loss_s * factor + loss_t
                    total_val_loss += loss
                va = accuracy_score(vlabels,np.where(np.array(vpredictions) < 0.5, 0.0,1.0))
                validation_accs = np.append(validation_accs,va)
            validation_losses = np.append(validation_losses,total_val_loss.detach().cpu().item())
            print("Epoch " + str(epoch) + " with val loss " + str(validation_losses[-1]) + " and val accuracy " + str(validation_accs[-1]))
            stop = es.earlyStopping(total_val_loss,model)
            #EARLY STOPPING
            if stop:
                print("TRAINING FINISHED AFTER " + str(epoch) + " EPOCHS. K BYE.")
                break
            else:
                if save:
                    np.save("vae_earlystopsave_training_losses.npy",training_losses)
                    np.save("vae_earlystopsave_mse_losses.npy",mse_losses)
                    np.save("vae_earlystopsave_kld_losses.npy",kld_losses)
                    np.save("vae_earlystopsave_siamese_losses.npy",siamese_losses)
                    np.save("vae_earlystopsave_validation_losses.npy",validation_losses)
                    np.save("vae_earlystopsave_validation_accs.npy",validation_accs)
                    np.save("vae_current_epoch.npy",np.array([epoch]))
    if (int(stop_epoch/10)-10) < len(validation_losses):
        final_loss = validation_losses[int(stop_epoch/10)-10] #10 coz of patience = 10
        final_acc = validation_accs[int(stop_epoch/10)-10]
    else:
        final_loss = validation_losses[-1]
    #SAVE LOSSES TO FILE
    if save == False or save == True:
        filename = str("AE_losses_siamese_simple_v2.txt")
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
Evaluates a network at the network_path (default none), given the test set at filepath and beta.
Prints the total loss of the test set in the console.
"""
def evalSet(filepath,beta,network_path=None):
    #get data
    loader, v_loader = getDatasets(filepath,8,validation_split=0,reduction=1,raw=False,augment=False)
    #get model
    if network_path:
        model = torch.load(network_path)
    else:
        model = facenetVAE()
    #calculate loss
    total_loss = 0
    total_mse = 0
    total_kld = 0
    model.eval()
    for inputs in loader:
        inputs = inputs.float().cuda()
        outputs,mu,logvar = model(inputs)
        lost,mselost,kldlost  = loss_fn(outputs,inputs,mu,logvar,beta)
        total_loss += lost.detach().cpu().item()
        total_mse += mselost.detach().cpu().item()
        total_kld += kldlost.detach().cpu().item()
    print("Total loss for testset is: " + str(total_loss) + " MSE loss: " + str(total_mse) + " KLD loss: " + str(total_kld))

"""
Loads the siamese net at the net_path (.pth file) and calculates the top 10 accuracy
of the test set te_enc_path and te_ids_path (both .npy files) 
based on the training set tr_enc_path and tr_ids_path (both .npy files)
Returns (and prints) the top 10 accuracy of the test set
"""
def top10Siamese(net_path,tr_path,te_path):
    batch_size=8
    #train_loader,val_loader,train_set,val_set = getDatasets(te_path,batch_size,raw=True,findDoubles=True)
    train_loader,val_loader,train_set,val_set = getDatasets(tr_path,1,validation_split=0,reduction=1,raw=True,augment=False,findDoubles=False,include_unlabelled=False)
    test_loader,val_loader,test_set,val_set = getDatasets(te_path,1,validation_split=0,reduction=1,raw=True,augment=False,findDoubles=False,include_unlabelled=False)

    pred = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    model = torch.load(net_path).cuda()
    model.eval()
    tr_len = train_set.getDatasetSize()
    te_len = test_set.getDatasetSize()
    for i in range(te_len):
      te,img_name,crop,te_id= test_set.getImageAndAll(i)#torch.from_numpy(np.array([te_enc[i],te_enc[i]])).float().cuda()
      matches = []
      te = te.float().cuda()
      for j in range(tr_len):
        tr,img_name,crop,tr_id = train_set.getImageAndAll(j)#torch.from_numpy(np.array([tr_enc[j],te_enc[i]])).float().cuda()
        tr = tr.float().cuda()
        r = model.forward_siamese(te,tr)
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
    model = facenetVAE().cuda()
    es = EarlyStopper(10,0.1,str("VAE_earlystopsave_siamese_simple_v2.pth"),False)
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
            o1,mu1,log1 = model.forward(b1)
            loss_t,_,_ = loss_fn(o1,b1,mu1,log1,beta)
            loss = loss_s * factor + loss_t
            loss.backward()
            optimizer.step()
            siamese_loss += loss_s
            total_train_loss += loss
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
                for b in range(int(math.ceil(val_set.getDatasetSize()/batch_size))):
                    b1,b2,l = val_set.getDoubleBatch(batch_size,b)
                    o = model.forward_siamese(b1,b2)
                    vlabels.extend(l[:,0].tolist())
                    vpredictions.extend(o[:,0].tolist())
                    loss_s = loss_siam(o,l)
                    o1,vmu,vlog = model.forward(b1)
                    loss_t,_,_ = loss_fn(o1,b1,vmu,vlog,beta)
                    loss = loss_s * factor + loss_t
                    total_val_loss += loss
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
    filename = str("vae_optim_v2.txt")
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
def optimal_optimisation():
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
