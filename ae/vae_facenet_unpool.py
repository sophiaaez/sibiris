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

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
import optuna
import random
from skimage import io, util
import skimage.transform 
import time


torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0),256,16,16) #normalerweise 256,16,16

class facenet(nn.Module):
    def __init__(self,channels=1,isize=512,z_dim=64):
        super(facenet,self).__init__()
        inplace = False
        h_dim = int(pow((isize/32),2)*256) #fuer ganzes netz 32 und 256 
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

def init_weights(m): #intialises weights with normal distribution over entire network
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight,gain=nn.init.calculate_gain('relu'))


def get_train_loader(batch_size,train_set):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=2,shuffle=True)
    return(train_loader)


def loss_fn(recon_x, x, mu, logvar):   # defining loss function for va-AE (loss= reconstruction loss + KLD (to analyse if we have normal distributon))
    a = torch.min(recon_x)
    b = torch.max(recon_x)
    c =torch.min(x)
    d =torch.max(x)
    if (a < 0.) or (b > 1.) or (c < 0.) or (d > 1.) or math.isnan(a) or math.isnan(b) or math.isnan(c) or math.isnan(d):
        print("STOPSTOPSTOPSTOPSTOP")
        print(str(a) + " " + str(b)+ " " + str(c)+ " " + str(d))
        return 0,0,0
    else:
        
        x = x.float().cuda()/256.
        BCE = F.binary_cross_entropy(recon_x, x)
        # source: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # KLD is equal to 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD, BCE, KLD


def trainNet(net, batch_size, n_epochs, learning_rate,train_loader,val_loader):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    net = net.float().cuda()
    #Get training data
    #train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=2,shuffle=True)
    n_batches = len(train_loader)
    
    #Create our loss and optimizer functions
    optimizer= optim.Adam(net.parameters(), lr=learning_rate)
    #loss, optimizer = createLossAndOptimizer(net, learning_rate)
    
    #Time for printing
    training_start_time = time.time()

    #Save Training and Validation Loss over time
    training_losses = []
    validation_losses = []
    
    #Loop for n_epochs
    for epoch in range(n_epochs):
        
        #running_loss = 0.0
        #print_every = n_batches // 10
        #start_time = time.time()
        total_train_loss = 0
        
        for i, inputs in enumerate(train_loader):
            #Get inputs
            #inputs = data.clone()
            #inputs = inputs.float().cuda()      
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs, mu, logvar = net(inputs)
            #outputs = outputs.float().cuda()
            #mu = mu.float().cuda()
            #logvar = logvar.float().cuda()
            #print(inputs.shape)
            #print(outputs.shape)
            BCEKLD, BCE, KLD = loss_fn(outputs, inputs,mu,logvar) # labels)
            BCEKLD.backward() #.mean().backward()
            optimizer.step()
            #return outputs
            #plt.imshow(outputs[0,:,:,0],cmap='gray')
            #Print statistics
            #running_loss += BCEKLD
            total_train_loss += BCEKLD
            
            """#Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                #print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                print(total_train_loss)
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()"""
            
        #At the end of the epoch, do a pass on the validation set
        with torch.no_grad():
            total_val_loss = 0
            for i, vinputs in enumerate(val_loader):
                #Forward pass
                val_outputs,vmu,vlogvar = net(vinputs)
                vBCEKLD, vBCE, vKLD  = loss_fn(val_outputs, inputs,vmu,vlogvar)
                total_val_loss += vBCEKLD
        training_losses.append(total_train_loss.detach().cpu().item()) 
        validation_losses.append(total_val_loss.detach().cpu().item())
        print("Eppch " +str(epoch))
        print("Training loss = {:.2f}".format(total_train_loss / len(train_loader)))   
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        if epoch%10 == 0:
                torch.save(net,str("cnnsave_" + str(epoch)))
                #Save validation and training losses to file 
                filename = str("losses" + str(time.time()) + ".txt")
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
        #Early Stopping
        """if validation_losses[:-1] > validation_losses[:-2]:
                torch.save(net,str("cnnsave_" + str(epoch)))
                #Save validation and training losses to file 
                filename = str("losses" + str(time.time()) + ".txt")
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
                break;    """
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))



class WhaleDataset(Dataset):
  def __init__(self,imagelist,transform=None):
    self.transform = transform
    self.imagelist = imagelist

  def __len__(self):
    return(len(self.imagelist))
  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    img_name = self.imagelist[idx]
    sizew = 512#512
    sizeh = 512#1024
    image = io.imread(img_name,as_gray=True)
    x,y = image.shape
    factorx = sizew/x
    factory = sizeh/y
    if factorx < factory:
          factor = factorx
    else:
          factor = factory
    im_rescaled = skimage.transform.rescale(image, factor, anti_aliasing=False)
    x,y =im_rescaled.shape
    x = (sizew-x)/2.0
    y = (sizeh-y)/2.0
    if int(x) < x:
          xx = int(x+1)
    else:
          xx = int(x)
    if int(y) < y:
          yy = int(y+1)
    else:
          yy = int(y)
    sample = util.pad(im_rescaled,((int(x),xx),(int(y),yy)),'edge')
    if self.transform:
      sample=self.transform(sample)
    return sample #.float().cuda()

#todo split train and val sets
def doYaThang(path,epochs,batch_size,load=""):
    transform = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #train_dir = f'{path+str("train/")}'
    #train_dataset = datasets.ImageFolder(train_dir,transform=transform)
    #dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=False, num_workers=8)
    imagelist = glob.glob(path+str("*.jpg"))
    #print(imagelist)
    random.shuffle(imagelist)
    validation_split = 0.33
    validation_amount = int(len(imagelist)*validation_split)
    validation_set = imagelist[:validation_amount]
    training_set = imagelist[validation_amount:]
    whales = WhaleDataset(imagelist=training_set,transform=transform)
    vhales = WhaleDataset(imagelist=validation_set,transform=transform)
    train_loader = torch.utils.data.DataLoader(whales, batch_size=batch_size,  num_workers=2,shuffle=True)
    val_loader = torch.utils.data.DataLoader(vhales, batch_size=batch_size,  num_workers=2,shuffle=True)
    #val_dir = f'{path+str("val/")}'
    #val_dataset = datasets.ImageFolder(val_dir,transform=transform)
    #val_dataloader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False, num_workers=8)
    
    if load:
        CNN = torch.load("cnnsave_" + load)
    else:
        CNN = facenet()
    trainNet(CNN, batch_size=batch_size, n_epochs=epochs, learning_rate=0.0001,train_loader=train_loader,val_loader=val_loader)
	#torch.save(CNN, str("cnnsave"))

def showOutputs(number):
	transform = transforms.Compose([transforms.ToTensor()])
	CNN = torch.load(str("trainV1_/cnnsave_"+str(number)))
	CNN = CNN.float().cuda()
	imagelist = glob.glob("test_images/*")
	vhales = WhaleDataset(imagelist,transform=transform)
	val_loader = torch.utils.data.DataLoader(vhales, batch_size=1,  num_workers=2)
	outputs = []
	for i, data in enumerate(val_loader, 0):
		output,mu,logvar = CNN(data.float().cuda())
		print(output[0,0].size())
		cv2.imwrite("test_images/output" + str(i) + ".jpg", output[0,0].cpu().detach().numpy())

def objective(trial):
    learning_rate =trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    batch_size = trial.suggest_int("batch_size",16,256,16)

def main():
	torch.cuda.set_device(0)
	doYaThang(path="../data/train/crops/",epochs=20,batch_size=16) #train/crops/*.jpg",epochs=2000)
	#showOutputs(200)
	

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
