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
import datetime

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

class facenetAE(nn.Module):
    def __init__(self,layer_amount=6,channels=1,isize=512,layer_size=128,extradense=False):
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

        #Decoder        

        self.t_conv4 = nn.ConvTranspose2d(256,384,2,stride=2)
        self.t_conv3 = nn.ConvTranspose2d(384,192,2,stride=2)
        self.t_conv2 = nn.ConvTranspose2d(192,64,2,stride=2)
        self.t_conv1 = nn.ConvTranspose2d(64, 1, 4, stride=4)

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

    def encode(self,x):
        x = nn.ReLU()(self.conv1(x))
        x = self.pool(x)
        x = self.lrn(x)
        #print(x.size())
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv2a(x))
        x = self.lrn(x)
        x = self.pool(x)
        #print(x.size())
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
        #print(x.size())
        h=self.fl(x)
        if self.extradense:
            h = self.extra(h)
        return(h)

    def forward(self, x):
        h = self.encode(x)
        x=self.unfl(h)
        if(self.layer_size == 128):
            x = nn.ReLU()(self.t_conv41(x))
        elif(self.layer_size == 64):
            x = nn.ReLU()(self.t_conv42(x))
        elif(self.layer_size == 32):
            x = nn.ReLU()(self.t_conv43(x))
        #print(x.size()) #[layer_size,16,16]
        x = nn.ReLU()(self.t_conv4(x))
        #print(x.size()) #[384,32,32]
        x = nn.ReLU()(self.t_conv3(x))
        #print(x.size()) #[192,64,64]
        x = nn.ReLU()(self.t_conv2(x))
        #print(x.size()) #[64,128,128]
        x = self.t_conv1(x)
        #print(x.size()) #[1,512,512]
        x = nn.Sigmoid()(x)
        #print(x.size())
        return x

def loss_fn(recon_x, x):   # defining loss function for va-AE (loss= reconstruction loss + KLD (to analyse if we have normal distributon))
    """a = round(torch.min(recon_x).item(),5)
    b = round(torch.max(recon_x).item(),5)
    c = round(torch.min(x).item(),5)
    d = round(torch.max(x).item(),5)
    #if (a < 0.) or (b > 1.) or (c < 0.) or (d > 1.) or math.isnan(a) or math.isnan(b) or math.isnan(c) or math.isnan(d):
        
    if (a >= 0.) and (b <= 1.) and (c >= 0.) and (d <= 1.) and not (math.isnan(a) or math.isnan(b) or math.isnan(c) or math.isnan(d)):
        BCE = F.binary_cross_entropy(recon_x, x)
        return BCE
    else:
        print("STOPSTOPSTOPSTOPSTOP")
        print("a" + str(a) + "b" + str(b) + "c" + str(c) + "d" + str(d))
        return 0,0,0"""
    loss = nn.MSELoss()
    l = loss(recon_x,x)
    #loss = F.mse_loss(recon_x, x, reduction='sum')
    return l

def trainNet(epochs,learning_rate,batch_size,data_path,layers,layer_size,save=True):
    now = datetime.datetime.now()
    #print(str(now))
    train_loader,val_loader = getDatasets(data_path,batch_size,reduction=1)
    model = facenetAE(layer_amount=layers,layer_size=layer_size).cuda()
    es = EarlyStopper(10,0.1,str("AE_earlystopsave_4_simple_v2_3.pth"),save)
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
            outputs = model(inputs)
            lost = loss_fn(outputs, inputs)
            lost.backward() 
            optimizer.step()
            total_train_loss += lost
        print("train loss " + str(total_train_loss.detach().cpu().item()))
        now = datetime.datetime.now()
        #print(str(now))
        training_losses.append(total_train_loss.detach().cpu().item()) 
        stop_epoch = epoch
        #VALIDATION 
        if epoch % 10 == 0: 
            model.eval()
            with torch.no_grad():
                total_val_loss = 0
                for vinputs in val_loader:
                    vinputs = vinputs.float().cuda()
                    val_outputs = model(vinputs)
                    vlost  = loss_fn(val_outputs,vinputs)
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
        filename = str("AE_losses_" + str(layers)+ "_" + str(layer_size) +"_simple_v2_3.txt")
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

def evalSet(filepath,network_path=None):
    #get data
    loader, v_loader = getDatasets(filepath,8,validation_split=0,reduction=1,raw=False,augment=False)
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
        outputs = model(inputs)
        lost  = loss_fn(outputs,inputs)
        total_loss += lost.detach().cpu().item()
    print("Total loss for testset is: " + str(total_loss))

def objective(trial):
    epochs = 1000
    data_path="../data/trainingset_final_v2.csv"
    #learning_rate =trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    #batch_size = trial.suggest_int("batch_size",8,32,8)
    learning_rate = 0.0001
    batch_size = 8
    layer_amount = trial.suggest_int("layer_amount",4,6,1)
    layer_size = trial.suggest_categorical("layer_size",[32,64,128])#,256])
    extradense = False #trial.suggest_categorical("extradense",[True,False])
    #print("BATCH SIZE: " + str(batch_size))
    print("Layer amount: " + str(layer_amount))
    print("Layer size: " + str(layer_size))
    print("Extra dense: " + str(extradense))
    train_loader,val_loader = getDatasets(data_path,batch_size)
    model = facenetAE(layer_amount=layer_amount,layer_size=layer_size,extradense=extradense).cuda()
    es = EarlyStopper(10,0.1,str("AE_earlystopsave_4_simple_v2.pth"),False)
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
            outputs = model(inputs)
            lost = loss_fn(outputs, inputs)
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
                for vinputs in val_loader:
                    vinputs = vinputs.float().cuda()
                    val_outputs = model(vinputs)
                    vlost  = loss_fn(val_outputs,vinputs)
                    total_val_loss += vlost
            validation_losses.append(total_val_loss.detach().cpu().item())
            print("Epoch " + str(epoch) + " with val loss " + str(validation_losses[-1]))
            stop = es.earlyStopping(total_val_loss,model)
            #EARLY STOPPING
            if stop:
                print("TRAINING FINISHED AFTER " + str(epoch) + " EPOCHS. K BYE.")
                break
            trial.report(total_val_loss,epoch)
    if (int(stop_epoch/10)-10) < len(validation_losses):
        final_loss = validation_losses[int(stop_epoch/10)-10] #10 coz of patience = 10
    else:
        final_loss = validation_losses[-1]
    #WRITE OPTIM 
    filename = str("ae_optim_v2.txt")
    file=open(filename,'a')
    file.write("layer_amount:" + str(layer_amount))
    file.write("layer_size:" + str(layer_size))
    file.write("extradense:" + str(extradense))
    file.write("final_loss:" + str(final_loss)) 
    file.write('\n') 
    file.close()
    return final_loss


def optimal_optimisation():
    torch.cuda.set_device(0)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=2)

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

if __name__ == "__main__":
    torch.cuda.empty_cache()
    trainNet(epochs=20,learning_rate=0.0001,batch_size=8,data_path="../data/trainingset_final_v2.csv",layers=4,layer_size=32,save=False)
    