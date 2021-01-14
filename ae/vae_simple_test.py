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

class facenetVAE(nn.Module):
    def __init__(self,
                 in_channels: int=1,
                 latent_dim: int=2000,
                 hidden_dims = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(facenetVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 32]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        number = 16*16
        self.fc_mu = nn.Linear(hidden_dims[-1]*number, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*number, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * number)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

        
    def reparameterize(self, mu, logvar):  # producing latent layer (Guassian distribution )
        std = torch.exp(0.5 * logvar) #logvar.mul(0.5).exp_()       # hint: var=std^2
        esp = torch.randn_like(std)#torch.randn(*mu.size()).cuda()   # normal unit distribution in shape of mu
        z = mu + std * esp     # mu:mean  std: standard deviation
        return z

    def encode(self,input):
        result = self.encoder(input)
        #print(result.size())
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        #print(result.size())
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 32,16,16)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        z = self.decode(z)
        #print(z.size())
        return  [z, mu, log_var]

    

def loss_fn(recon_x, x,mu,logvar,beta):   # defining loss function for va-AE (loss= reconstruction loss + KLD (to analyse if we have normal distributon))
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
    kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
    return l+(beta*kld),l,kld

def trainNet(epochs,learning_rate,batch_size,data_path,layers,layer_size,beta=1,save=True):
    train_loader,val_loader = getDatasets(data_path,batch_size,reduction=0.25)
    model = facenetVAE().cuda()
    es = EarlyStopper(10,0.1,str("VAE_earlystopsave_4_simple_v2_zdim.pth"),save)
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
        filename = str("VAE_losses_" + str(layers)+ "_" + str(layer_size) +"_simple_2_zdim.txt")
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

def objective(trial):
    epochs = 1000
    data_path="../data/trainingset_final_v2.csv"
    #learning_rate =trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    #batch_size = trial.suggest_int("batch_size",8,32,8)
    learning_rate = 0.0001
    batch_size = 8
    layer_amount = 4 #trial.suggest_int("layer_amount",4,6,1)
    layer_size = trial.suggest_categorical("layer_size",[32,128])#,256])
    extradense = False #trial.suggest_categorical("extradense",[True,False])
    #beta = trial.suggest_int("beta",1,20,1)
    beta = 1
    #print("BATCH SIZE: " + str(batch_size))
    #print("Layer amount: " + str(layer_amount))
    #print("Layer size: " + str(layer_size))
    #print("extradense: " + str(extradense))
    print("beta: " + str(beta))
    train_loader,val_loader = getDatasets(data_path,batch_size)
    model = facenetVAE(layer_amount=layer_amount,layer_size=layer_size,extradense=extradense).cuda()
    es = EarlyStopper(10,0.1,str("AE_earlystopsave_4_simple_v2.pth"),False)
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
            stop = es.earlyStopping(total_val_loss,model)
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

def objective_small(trial):
    epochs = 1000
    data_path="../data/trainingset_final_v2.csv"
    #learning_rate =trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    #batch_size = trial.suggest_int("batch_size",8,32,8)
    learning_rate = 0.0001
    batch_size = 8
    layer_amount = 4#trial.suggest_int("layer_amount",4,6,1)
    layer_size = 128#trial.suggest_categorical("layer_size",[64,128])#,256])
    beta = trial.suggest_int("beta",1,20,1)
    extradense = False # trial.suggest_categorical("extradense",[True,False])
    #print("BATCH SIZE: " + str(batch_size))
    #print("Layer amount: " + str(layer_amount))
    #print("Layer size: " + str(layer_size))
    #print("extradense: " + str(extradense))
    print("beta: " + str(beta))
    train_loader,val_loader = getDatasets(data_path,batch_size,reduction=0.25)
    model = facenetVAE(layer_amount=layer_amount,layer_size=layer_size,extradense=extradense).cuda()
    es = EarlyStopper(10,0.1,str("AE_earlystopsave_4_simple_v2.pth"),False)
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
            stop = es.earlyStopping(total_val_loss,model)
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
    filename = str("vae_optim_v2_beta.txt")
    file=open(filename,'a')
    file.write("beta:" + str(beta))
    file.write("final_loss:" + str(final_loss)) 
    file.write("final_mse_loss:" + str(final_mse_loss)) 
    file.write('\n')
    file.close()
    return final_loss


def optimal_optimisation():
    torch.cuda.set_device(0)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=4)

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


def optimal_optimisation_small():
    torch.cuda.set_device(0)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_small,n_trials=20)

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
    