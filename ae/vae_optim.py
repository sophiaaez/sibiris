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

import glob
import numpy as np
import matplotlib.pyplot as plt
import math
import optuna
from PIL import Image
import random
from skimage import io, util, transform, color

from torch.utils.tensorboard import SummaryWriter


torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0),256,1,1) #normalerweise 256,16,16 

class facenet(nn.Module):
    def __init__(self,channels=1,isize=512,z_dim=64):
        super(facenet,self).__init__()
        inplace = False
        h_dim = 256#int(pow((isize/32),2)*256) #fuer ganzes netz 32 und 256 
        self.cv1 = nn.Sequential(#input 1 500 1000
            nn.Conv2d(channels, 64, 7, stride=2, padding=3),  #64, 256,512
            nn.ReLU(inplace),
        )
        self.mp1 = nn.MaxPool2d(3,stride=2,padding=1, return_indices=True)  #64, 128,256
        self.cv2 = nn.Sequential(
                nn.LocalResponseNorm(2),
                nn.Conv2d(64, 64,1,stride=1), #64, 128,256
                nn.ReLU(inplace),
                nn.Conv2d(64, 192,3,stride=1,padding=1), #192, 128, 256
                nn.ReLU(inplace),
                nn.LocalResponseNorm(2),
        )
        self.mp2 = nn.MaxPool2d(3,stride=2,padding=1, return_indices=True) #192, 64,128 
        self.cv3 = nn.Sequential(
                nn.Conv2d(192, 192,1,stride=1), #192, 64, 128
                nn.ReLU(inplace),
                nn.Conv2d(192,384,3,stride=1,padding=1), #384, 64,128
                nn.ReLU(inplace), 
        )
        self.mp3 = nn.MaxPool2d(3,stride=2,padding=1, return_indices=True) #384, 32, 64
        self.cv4 = nn.Sequential(
                nn.Conv2d(384,384,1,stride=1), #384, 32, 64
                nn.ReLU(inplace),
                nn.Conv2d(384,256,3,stride=1,padding=1), #256, 32, 64
                nn.ReLU(inplace),
        )
        self.cv5 = nn.Sequential(
                nn.Conv2d(256,256,1,stride=1), #256, 32, 64
                nn.ReLU(inplace),
                nn.Conv2d(256,256,3,stride=1,padding=1), #256, 32, 64
                nn.ReLU(inplace),
        )
        self.cv6 = nn.Sequential(
                nn.Conv2d(256,256,1,stride=1), #256, 32, 64
                nn.ReLU(inplace),
                nn.Conv2d(256,256,3,stride=1,padding=1), #256, 32, 64
                nn.ReLU(inplace),
        )
        self.mp6 = nn.MaxPool2d(3,stride=2,padding=1, return_indices=True) #256, 16, 32

        self.fl = Flatten()
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.unfl = UnFlatten()
        self.up6 = nn.MaxUnpool2d(3,stride=2,padding=1)
        
        self.cvt6 = nn.Sequential(# 256,16,16
                nn.ConvTranspose2d(256,256,3,padding=1, stride=1), #256, 32,32
                nn.ReLU(inplace),
                nn.ConvTranspose2d(256,256,1,stride=1), #10, 256, 32,32
                nn.ReLU(inplace),
        )
        self.cvt5 = nn.Sequential(
                nn.ConvTranspose2d(256,256,3, stride=1,padding=1), #256, 32,32
                nn.ReLU(inplace),
                nn.ConvTranspose2d(256,256,1, stride=1), #256, 32,32
                nn.ReLU(inplace),
        )
        self.cvt4 = nn.Sequential(
                nn.ConvTranspose2d(256,384,3, stride=1,padding=1), #384, 32,32
                nn.ReLU(inplace),
                nn.ConvTranspose2d(384,384,1, stride=1), #384, 32,32
                nn.ReLU(inplace),
        )
        
        self.up3 = nn.MaxUnpool2d(3,stride=2,padding=1)
        self.cvt3 = nn.Sequential(
                nn.ConvTranspose2d(384,192,3, stride=1,padding=1),#,padding=(0,1)), #192, 64,64
                nn.ReLU(inplace),
                nn.ConvTranspose2d(192,192,1, stride=1), #192, 64,64
                nn.ReLU(inplace), 
                nn.BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True),
        )
        self.up2 = nn.MaxUnpool2d(3,stride=2,padding=1)
        self.cvt2 = nn.Sequential(
                nn.ConvTranspose2d(192,64,3, stride=1,padding=1), #64, 128,128
                nn.ReLU(inplace),
                nn.ConvTranspose2d(64,64,1,stride=1), #64, 128,128
                nn.ReLU(inplace),                
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
        )
        self.up1 = nn.MaxUnpool2d(3,stride=2,padding=1)
        self.cvt1 = nn.Sequential(
                nn.ConvTranspose2d(64,channels,8, stride=2, padding=3,output_padding=0, dilation=1, padding_mode='zeros'),#3,512,512
                nn.ReLU(inplace),
                nn.Sigmoid()
                #nn.Tanh()
        )
        networkywork = nn.Sequential(
            self.cv1,
            self.cv2,
            self.cv3,
            self.cv4,
            self.cv5,
            self.cv6,
            self.fc1,
            self.fc2,
            self.fc3,
            self.cvt6,
            self.cvt5,
            self.cvt4,
            self.cvt3,
            self.cvt2,
            self.cvt1
        )   
        networkywork.apply(init_weights)
        
    def reparameterize(self, mu, logvar):  # producing latent layer (Guassian distribution )
        std = logvar.mul(0.5).exp_()       # hint: var=std^2
        esp = torch.randn(*mu.size()).cuda()   # normal unit distribution in shape of mu
        z = mu + std * esp     # mu:mean  std: standard deviation
        return z
    
    def bottleneck(self, h):      # hidden layer ---> mean layer + logvar layer
        mu = self.fc1(h)
        logvar = self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


    def forward(self, x):
        #encoding
        out = self.cv1(x.float().cuda())
        size1=out.size()
        out,i1=self.mp1(out)
        out=self.cv2(out)
        size2=out.size()
        out,i2=self.mp2(out)
        out=self.cv3(out)
        size3=out.size()
        out,i3=self.mp3(out)
        out=self.cv4(out)
        out=self.cv5(out)
        out=self.cv6(out)
        size6=out.size()
        out,i6=self.mp6(out)
        #bottlenecking
        h=self.fl(out)
        z,mu,logvar=self.bottleneck(h)
        h=self.fc3(z)
        out=self.unfl(h)
        #decoding
        out=self.up6(out,i6,output_size=size6)
        out=self.cvt6(out)
        out=self.cvt5(out)
        out=self.cvt4(out)
        out=self.up3(out,i3,output_size=size3)
        out=self.cvt3(out)
        out=self.up2(out,i2,output_size=size2)
        out=self.cvt2(out)
        out=self.up1(out,i1,output_size=size1)
        out=self.cvt1(out)
        return(out, mu, logvar) 

    """def encode(self,x):
        out = self.cv1(x.float().cuda())
        size1=out.size()
        out,i1=self.mp1(out)
        out=self.cv2(out)
        size2=out.size()
        out,i2=self.mp2(out)
        out=self.cv3(out)
        size3=out.size()
        out,i3=self.mp3(out)
        out=self.cv4(out)
        out=self.cv5(out)
        out=self.cv6(out)
        size6=out.size()
        out,i6=self.mp6(out)
        #bottlenecking
        h=self.fl(out)
        return out"""

def init_weights(m): #intialises weights with normal distribution over entire network
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight,gain=nn.init.calculate_gain('relu'))

def loss_fn(recon_x, x, mu, logvar):   # defining loss function for va-AE (loss= reconstruction loss + KLD (to analyse if we have normal distributon))
    a = round(torch.min(recon_x).item(),5)
    b = round(torch.max(recon_x).item(),5)
    c = round(torch.min(x).item(),5)
    d = round(torch.max(x).item(),5)
    #if (a < 0.) or (b > 1.) or (c < 0.) or (d > 1.) or math.isnan(a) or math.isnan(b) or math.isnan(c) or math.isnan(d):
        
    if (a >= 0.) and (b <= 1.) and (c >= 0.) and (d <= 1.) and not (math.isnan(a) or math.isnan(b) or math.isnan(c) or math.isnan(d)):
        x = x.float().cuda()/256.
        BCE = F.binary_cross_entropy(recon_x, x)
        #BCEL = nn.BCEWithLogitsLoss(recon_x, x)
        # source: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # KLD is equal to 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD, BCE, KLD
    else:
        print("STOPSTOPSTOPSTOPSTOP")
        print("a" + str(a) + "b" + str(b) + "c" + str(c) + "d" + str(d))
        return 0,0,0

class WhaleDataset(Dataset):
  def __init__(self,imagelist,img_size):
    self.transform = transform
    self.imagelist = imagelist
    self.img_size = img_size

  def __len__(self):
    return(len(self.imagelist))
  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    img_name = self.imagelist[idx]
    image = io.imread(img_name,as_gray=True)
    image = transform.resize(image,(self.img_size,self.img_size))
    image = transforms.ToTensor()(image)
    return image

  def getImageAndName(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    img_name = self.imagelist[idx]
    image = io.imread(img_name,as_gray=True)
    image = transform.resize(image,(self.img_size,self.img_size))
    image = torch.from_numpy(np.array([[image],[image]]))
    #image = transforms.ToTensor()(image)
    return image,img_name
    
  def getNameForIDX(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    img_name = self.imagelist[idx]
    return img_name

def getDatasets(path,batch_size,validation_split=1/3):
    size = 512
    imagelist = glob.glob(path+str("*.jpg"))
    random.shuffle(imagelist)
    validation_amount = int(len(imagelist)*validation_split)
    validation_set = imagelist[:validation_amount]
    training_set = imagelist[validation_amount:]
    whales = WhaleDataset(imagelist=training_set,img_size=size)
    vhales = WhaleDataset(imagelist=validation_set,img_size=size)
    train_loader = torch.utils.data.DataLoader(whales, batch_size=batch_size,  num_workers=2,shuffle=True)
    val_loader = torch.utils.data.DataLoader(vhales, batch_size=batch_size,  num_workers=2,shuffle=True)
    return train_loader, val_loader

def getMISTDatasets(batch_size):
    transformer = transforms.Compose([transforms.ToTensor()])
    trainset=datasets.MNIST(root='data', train=True, download=True, transform=transformer)
    testset=datasets.MNIST(root='data', train=False, download=True, transform=transformer)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=0)
    return train_loader, test_loader


def trainNet(epochs,learning_rate,batch_size,save,data_path):
    #writer = SummaryWriter('whales')
    #DATA
    train_loader,val_loader = getMISTDatasets(batch_size) #../data/train/crops/",batch_size)
    #dataiter = iter(train_loader)
    #images = dataiter.next()
    #MODEL
    model = facenet().cuda()
    #writer.add_graph(model,images)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #loss = nn.BCEWithLogitsLoss()
    model.train()
    #TRAINING
    training_losses = []
    validation_losses = []
    for epoch in range(epochs):
        total_train_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            #inputs = inputs.float().cuda()
            outputs, mu, logvar = model(inputs)
            BCEKLD, BCE, KLD = loss_fn(outputs, inputs,mu,logvar)
            #bcel = loss(outputs.float(),inputs.float())
            BCEKLD.backward() 
            optimizer.step()
            total_train_loss += BCEKLD
        print("train loss " + str(total_train_loss.detach().cpu().item()))
        #VALIDATION 
        if epoch % 10 == 0: 
            model.eval()
            with torch.no_grad():
                total_val_loss = 0
                for vinputs, labels in val_loader:
                    #vinputs = vinputs.float().cuda()
                    val_outputs,vmu,vlogvar = model(vinputs)
                    vBCEKLD, vBCE, vKLD  = loss_fn(val_outputs, vinputs,vmu,vlogvar)
                    total_val_loss += vBCEKLD
                    image  =val_outputs[0,0].cpu().detach()
                    #io.imsave("test_images/set/outputs/output_" + str(epoch) + ".jpg", (color.grey2rgb(image)*255).astype(np.uint8))
            model.train() #reset to train
            validation_losses.append(total_val_loss.detach().cpu().item())
            print("Epoch " + str(epoch) + " with val loss " + str(validation_losses[-1]))
            #EARLY STOPPING
            if False: #len(validation_losses) > 2 and ((validation_losses[-1] - validation_losses[-2]) > 0.1 or round(validation_losses[-3],5) == round(validation_losses[-1],5)):
                print("TRAINING FINISHED AFTER " + str(epoch) + " EPOCHS. K BYE.")
                break
            else:
                if save:
                    torch.save(model,str("modelsave_t_"+str(epoch)+"e")) #makes sure to save before validation loss increases
        training_losses.append(total_train_loss.detach().cpu().item()) 
    writer.close()
    #SAVE LOSSES TO FILE
    if save:
        filename = str("losses_t.txt")
        file=open(filename,'w')
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

def getAndSaveOutputs(image_path,network_path=None):
    imagelist = glob.glob(image_path + str("*.jpg"))
    dataset = WhaleDataset(imagelist,512)
    encoding_ids = None
    encodings = np.array([])
    if network_path:
        model = torch.load(network_path)
    else:
        model = facenet()
    for i in range(len(dataset)):
        img, img_name = dataset.getImageAndName(i)
        output,mu,logvar = model.forward(img)
        imagename = img_name.split("/")[-1]
        image  =output[0,0].cpu().detach()
        io.imsave(image_path + "outputs/" + imagename, (color.grey2rgb(image)*255).astype(np.uint8))
        print(image_path + "outputs/" + imagename)

def getAndSaveEncodings(image_path,network_path=None):
    imagelist = glob.glob(image_path + str("*.jpg"))
    dataset = WhaleDataset(imagelist,512)
    encoding_ids = None
    encodings = np.array([])
    if network_path:
        model = torch.load(network_path)
    else:
        model = facenet()
    for i in range(len(dataset)):
        img, img_name = dataset.getImageAndName(i)
        encoding = model.encode(img)
        print(i)
        eco=encoding[0,:,:,:].detach().cpu().numpy()
        if i == 0:
            encodings = np.array([eco])
            encoding_ids = np.array([img_name])
        else:
            encodings = np.append(encodings,[eco],axis=0)
            encoding_ids = np.append(encoding_ids,[img_name],axis=0)
    with open('encodings.npy', 'wb') as f:
        np.save(f, encodings)
    with open('encoding_ids.npy','wb') as f:
        np.save(f,encoding_ids)
    #return encodings



def objective(trial):
    epochs = 100
    learning_rate =trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_int("batch_size",8,32,8)
    print("BATCH SIZE: " + str(batch_size))
    #DATA
    train_loader,val_loader = getDatasets("../data/train/crops/",batch_size)
    #MODEL
    model = facenet()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    #TRAINING
    training_losses = []
    validation_losses = []
    final_loss = None
    for epoch in range(epochs):
        total_train_loss = 0
        for i, inputs in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.float().cuda()
            outputs, mu, logvar = model(inputs)
            BCEKLD, BCE, KLD = loss_fn(outputs, inputs,mu,logvar)
            BCEKLD.backward() 
            optimizer.step()
            total_train_loss += BCEKLD
        #VALIDATION 
        if epoch % 10 == 0: 
            model.eval()
            with torch.no_grad():
                total_val_loss = 0
                for i, vinputs in enumerate(val_loader):
                    val_outputs,vmu,vlogvar = model(vinputs)
                    vBCEKLD, vBCE, vKLD  = loss_fn(val_outputs, vinputs,vmu,vlogvar)
                    total_val_loss += vBCEKLD
            model.train() #reset to train
            validation_losses.append(total_val_loss.detach().cpu().item())
            #EARLY STOPPING
            if len(validation_losses) > 1 and validation_losses[-1] > validation_losses[-2]:
                print("TRAINING FINISHED AFTER " + str(epoch) + " EPOCHS. K BYE.")
                final_loss = training_losses[-10]
                break
            #else:
                #torch.save(model,str("modelsave")) #makes sure to save before validation loss increases
        training_losses.append(total_train_loss.detach().cpu().item()) 
        trial.report(total_train_loss,epoch)
    if not final_loss:
    	final_loss = training_losses[-1]
    #WRITE OPTIM 
    filename = str("optim.txt")
    file=open(filename,'w')
    file.write("batch_size:" + str(batch_size))
    file.write("learning_rate:" + str(learning_rate))
    file.write("final_loss:" + str(final_loss))  
    file.close()
    return final_loss


def main():
    torch.cuda.set_device(0)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=100)

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


def main2():
    getAndSaveEncodings("../data/train/crops/")

def main3():
    trainNet(epochs=300,learning_rate=0.001,batch_size=32,save=False,data_path="test_images/set/")

def main4():
	getAndSaveOutputs("test_images/set/","train_test_/modelsave_t_30e")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main3()
    
