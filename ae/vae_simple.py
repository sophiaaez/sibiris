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
#from torchsummary import summary

import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import random
from skimage import io, util, transform, color,exposure,filters
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
The facenet Variational Autoencoder module with the following parameters:
layer_amount = the amount of regular layers in the encoder, possible values 4,5,6, default 6
channels = the amount of channels of the input image, default 1
isize = the size of the square input image, default 512 for 512x512
layer_size = the amount of feature maps at the bottleneck, possible vlaues 32,64,128,256, default 32
extradense = an additional dense layer at the bottleneck is added if this parameter is set to True, default False
"""
class facenetVAE(nn.Module):
    def __init__(self,layer_amount=6,channels=1,isize=512,layer_size=32,extradense=False):
        super(facenetVAE,self).__init__()
        poolamount = 16
        h_dim = int(pow((isize/(poolamount*2)),2)*layer_size)
        z_dim = int(h_dim/4)
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
        if self.extradense:
            self.extra = nn.Linear(h_dim,h_dim)
        self.fl = Flatten()
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

        self.fc3 = nn.Linear(z_dim, h_dim)
        self.unfl = UnFlatten(layer_size)

        #Decoder
        self.t_conv41 = nn.ConvTranspose2d(128,256,1,stride=1)
        self.t_conv42 = nn.ConvTranspose2d(64,256,1,stride=1)
        self.t_conv43 = nn.ConvTranspose2d(32,256,1,stride=1)

        self.t_conv4 = nn.ConvTranspose2d(256,384,2,stride=2)
        self.t_conv3 = nn.ConvTranspose2d(384,192,2,stride=2)
        self.t_conv2 = nn.ConvTranspose2d(192,64,2,stride=2)
        self.t_conv1 = nn.ConvTranspose2d(64, 1, 4, stride=4)

        #Weight Initialisation
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
    def reparameterize(self, mu, logvar):  
        std = torch.exp(0.5 * logvar) 
        esp = torch.randn_like(std)
        z = mu + torch.sqrt(std) * esp    
        return z
    
    """
    Calculates the mu and logvar for the latent vector h and resamples it from these vectors.
    Returns resampled vector z, mu and logvar.
    """
    def bottleneck(self, h):  
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
        if self.extradense:
            h = self.extra(h)
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
            x = nn.ReLU()(self.t_conv43(x))#[layer_size,16,16]
        x = nn.ReLU()(self.t_conv4(x))#[384,32,32]
        x = nn.ReLU()(self.t_conv3(x))#[192,64,64]
        x = nn.ReLU()(self.t_conv2(x))#[64,128,128]
        x = self.t_conv1(x)#[1,512,512]
        x = nn.Sigmoid()(x)
        return x,mu,logvar

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

"""
Creates and trains the facenetVAE network given the parameters:
epochs = amount of maximum epochs
learning_rate = the learning rate used to train the network
batch_size = the batch size used to train the network
data_path = the path to the training data set
layers = the amount of regular layers (possible values = 4,5,6)
layer_size = the amount of feature maps at the bottleneck (possible values = 32,64,128,256)
beta = the weight factor weighing the MSE and KLD losses against each other, default 1
save = a boolean whether the network should be saved or not, default True
"""
def trainNet(epochs,learning_rate,batch_size,data_path,layers,layer_size,beta=1,save=True):
    train_loader,val_loader = getDatasets(data_path,batch_size,reduction=1)
    model = facenetVAE(layer_amount=layers,layer_size=layer_size).cuda()
    es = EarlyStopper(10,0.1,str("VAE_earlystopsave_4_simple_v3_1.pth"),save)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #TRAINING
    training_losses = []
    validation_losses = []
    for epoch in range(epochs):
        total_train_loss = 0
        model.train()
        for inputs in train_loader:
            optimizer.zero_grad()
            inputs = inputs.float().cuda()
            outputs, mu, logvar = model(inputs)
            lost, mselost, kldlost = loss_fn(outputs, inputs,mu,logvar,beta)
            lost.backward() 
            optimizer.step()
            total_train_loss += lost
        print("train loss " + str(total_train_loss.detach().cpu().item()))
        training_losses.append(total_train_loss.detach().cpu().item()) 
        stop_epoch = epoch
        #VALIDATION 
        if epoch % 10 == 0: 
            model.eval()
            with torch.no_grad():
                total_val_loss = 0
                for vinputs in val_loader:
                    vinputs = vinputs.float().cuda()
                    voutputs, vmu, vlogvar = model(vinputs)
                    vlost, vmselost, vkldlost = loss_fn(voutputs, vinputs,vmu,vlogvar,beta)
                    total_val_loss += vlost
            validation_losses.append(total_val_loss.detach().cpu().item())
            print("Epoch " + str(epoch) + " with val loss " + str(validation_losses[-1]))
            stop = es.earlyStopping(total_val_loss,model)
            #EARLY STOPPING
            if stop:
                print("TRAINING FINISHED AFTER " + str(epoch) + " EPOCHS. K BYE.")
                break
    #SAVE LOSSES TO FILE
    if save:
        filename = str("VAE_losses_" + str(layers)+ "_" + str(layer_size) +"_simple_v3_1.txt")
        file=open(filename,'w')
        file.write("trained with learning rate " + str(learning_rate) + ", batch size " + str(batch_size) + ", planned epochs " + str(epochs) + " but only took " + str(stop_epoch) + " epochs.")
        file.write("training_losses")
        file.write('\n')
        for element in training_losses:
            file.write(str(element))
            file.write('\n')
        file.write("validation_losses")
        file.write('\n')
        for element in validation_losses:
            file.write(str(element))
            file.write('\n')   
        file.close()

"""
Evaluates a network at the network_path (default none), given the test set at filepath and the beta value beta.
Prints the total losses of the test set in the console.
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
The objective for the optimisation.
"""
def objective(trial):
    epochs = 1000
    data_path="../data/trainingset_final_v2.csv"
    #FINDING VALUES TO TEST IN THIS TRIAL
    learning_rate = 0.0001 #trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = 8 #trial.suggest_int("batch_size",8,32,8)
    layer_amount = trial.suggest_int("layer_amount",4,6,1)
    layer_size = trial.suggest_categorical("layer_size",[32,64,128])#,256])
    extradense = False #trial.suggest_categorical("extradense",[True,False])
    beta = trial.suggest_int("beta",1,20,1)
    print("Learning Rate: " + str(learning_rate))
    print("BATCH SIZE: " + str(batch_size))
    print("Layer amount: " + str(layer_amount))
    print("Layer size: " + str(layer_size))
    print("extradense: " + str(extradense))
    print("beta: " + str(beta))
    train_loader,val_loader = getDatasets(data_path,batch_size,reduction=0.25)
    model = facenetVAE(layer_amount=layer_amount,layer_size=layer_size,extradense=extradense).cuda()
    es = EarlyStopper(10,0.1,str("VAE_earlystopsave_4_simple_v3.pth"),False)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #TRAINING
    training_losses = []
    validation_losses = []
    val_mse = []
    for epoch in range(epochs):
        total_train_loss = 0
        model.train()
        for inputs in train_loader:
            optimizer.zero_grad()
            inputs = inputs.float().cuda()
            outputs, mu, logvar = model(inputs)
            lost, mselost, kldlost = loss_fn(outputs, inputs,mu,logvar,beta)
            lost.backward() 
            optimizer.step()
            total_train_loss += lost
        #print("train loss " + str(total_train_loss.detach().cpu().item()))
        training_losses.append(total_train_loss.detach().cpu().item()) 
        stop_epoch = epoch
        #VALIDATION 
        if epoch % 10 == 0: 
            model.eval()
            with torch.no_grad():
                total_val_loss = 0
                total_mse_loss = 0
                for vinputs in val_loader:
                    vinputs = vinputs.float().cuda()
                    voutputs, vmu, vlogvar = model(vinputs)
                    vlost, vmselost, vkldlost = loss_fn(voutputs, vinputs,vmu,vlogvar,beta)
                    total_val_loss += vlost
                    total_mse_loss += vmselost
            validation_losses.append(total_val_loss.detach().cpu().item())
            val_mse.append(total_mse_loss.detach().cpu().item())
            print("Epoch " + str(epoch) + " with val loss " + str(validation_losses[-1]))
            stop = es.earlyStopping(total_mse_loss,model)
            #EARLY STOPPING
            if stop:
                print("TRAINING FINISHED AFTER " + str(epoch) + " EPOCHS. K BYE.")
                break
            trial.report(total_val_loss,epoch)
    if (int(stop_epoch/10)-10) > 0 and (int(stop_epoch/10)-10) < len(validation_losses):
        final_loss = validation_losses[int(stop_epoch/10)-10] #10 coz of patience = 10
        final_mse_loss = val_mse[int(stop_epoch/10)-10]
    else:
        final_loss = validation_losses[-1]
        final_mse_loss = val_mse[-1]
    #WRITE OPTIM 
    filename = str("vae_optim_v2.txt")
    file=open(filename,'a')
    file.write("layer_amount:" + str(layer_amount))
    file.write("layer_size:" + str(layer_size))
    file.write("extradense:" + str(extradense))
    file.write("beta:" + str(beta))
    file.write("final_loss:" + str(final_loss)) 
    file.write("final_mse_loss:" + str(final_mse_loss)) 
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