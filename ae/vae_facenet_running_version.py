import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from skimage import io, util
import skimage.transform 
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import random

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0),256,16,16)

class facenet(nn.Module):
  def __init__(self,channels=1,h_dim=65536,z_dim=100):
    super(facenet,self).__init__()

    self.cv1 = nn.Sequential(#input 1 500 1000
            #Layer1
            nn.Conv2d(channels, 64, 7, stride=2, padding=3),  #64, 256,512
            nn.ReLU(True),
            nn.MaxPool2d(3,stride=2,padding=1),  #64, 128,256
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
    )
    self.cv2 = nn.Sequential(
        #Layer2
            nn.Conv2d(64, 64,1,stride=1), #64, 128,256
            nn.ReLU(True),
            nn.Conv2d(64, 192,3,stride=1,padding=1), #192, 128, 256
            nn.ReLU(True),
            nn.BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True),
            nn.MaxPool2d(3,stride=2,padding=1), #192, 64,128 
    )

    self.cv3 = nn.Sequential(
            nn.Conv2d(192, 192,1,stride=1), #192, 64, 128
            nn.ReLU(True),
            nn.Conv2d(192,384,3,stride=1,padding=1), #384, 64,128
            nn.ReLU(True), 
            nn.MaxPool2d(3,stride=2,padding=1), #384, 32, 64
    )
    self.cv4 = nn.Sequential(
            nn.Conv2d(384,384,1,stride=1), #384, 32, 64
            nn.ReLU(True),
            nn.Conv2d(384,256,3,stride=1,padding=1), #256, 32, 64
            nn.ReLU(True),
    )
    self.cv5 = nn.Sequential(
            nn.Conv2d(256,256,1,stride=1), #256, 32, 64
            nn.ReLU(True),
            nn.Conv2d(256,256,3,stride=1,padding=1), #256, 32, 64
            nn.ReLU(True),
    )
    self.cv6 = nn.Sequential(
            nn.Conv2d(256,256,1,stride=1), #256, 32, 64
            nn.ReLU(True),
            nn.Conv2d(256,256,3,stride=1,padding=1), #256, 32, 64
            nn.ReLU(True),
            nn.MaxPool2d(3,stride=2,padding=1), #256, 16, 32
    )
    self.fl = Flatten()
    #elf.encoder = nn.Sequential()
    
    self.fc1 = nn.Linear(h_dim, z_dim)
    self.fc2 = nn.Linear(h_dim, z_dim)
    self.fc3 = nn.Linear(z_dim, h_dim)

    self.unfl = UnFlatten()

    self.cvt6 = nn.Sequential(# 256,16,16
            nn.ConvTranspose2d(256,256,5,padding=2, stride=2,output_padding=1), #256, 32,32
            nn.ReLU(True),
            nn.ConvTranspose2d(256,256,3,padding=1,stride=1), #10, 256, 32,32
            nn.ReLU(True),
    )
    self.cvt5 = nn.Sequential(
            nn.ConvTranspose2d(256,256,3, stride=1,padding=1), #256, 32,32
            nn.ReLU(True),
            nn.ConvTranspose2d(256,256,1, stride=1), #256, 32,32
            nn.ReLU(True),
    )
    self.cvt4 = nn.Sequential(
            nn.ConvTranspose2d(256,384,3, stride=1,padding=1), #384, 32,32
            nn.ReLU(True),
            nn.ConvTranspose2d(384,384,1, stride=1), #384, 32,32
            nn.ReLU(True),
    )
    self.cvt3 = nn.Sequential(
            nn.ConvTranspose2d(384,192,5, stride=2,padding=2,output_padding=1),#,padding=(0,1)), #192, 64,64
            nn.ReLU(True),
            nn.ConvTranspose2d(192,192,1, stride=1), #192, 64,64
            nn.ReLU(True), 
    )
    self.cvt2 = nn.Sequential(
            nn.BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True),
            nn.ConvTranspose2d(192,64,5, stride=2,padding=2,output_padding=1), #64, 128,128
            nn.ReLU(True),
            nn.ConvTranspose2d(64,64,1, stride=1), #64, 128,128
            nn.ReLU(True),
    )
    self.cvt1 = nn.Sequential(
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ConvTranspose2d(64,channels,7, stride=4,padding=2,output_padding=1),#3,512,512
            nn.ReLU(True),
            nn.Tanh()
    )
    self.decoder = nn.Sequential(
            
           
            #Layer-2
            
            #Layer-1
            
            
    )

  def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

  def bottleneck(self, h):
        #print(h.shape)
        mu = self.fc1(h)
        logvar =  self.fc2(h)
        #print(mu.shape)
        #print(logvar.shape)
        z = self.reparameterize(mu, logvar)
        #print(z.shape)
        return z, mu, logvar

  def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

  def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return(z, mu, logvar)

  def encode(self,x):
        #print(x.shape)
        h = self.encoder(x)
        #print(h.shape)
        z,mu,logvar = self.bottleneck(h)
        return(z,mu,logvar)

  def decode(self,z):
        z = self.fc3(z)
        z = self.decoder(z)
        #print(z.shape)
        return(z)

def outputSize(in_size, kernel_size,stride,padding):
    output = int((in_size - kernel_size + 2*(padding))/stride)+1
    return(output)

def get_train_loader(batch_size,train_set):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=2,shuffle=True)
    return(train_loader)


def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x,x.detach(),reduction='mean')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
    return BCE + KLD

def createLossAndOptimizer(net, learning_rate=0.001):
    
    #Loss function
    loss = nn.BCELoss()
    
    #Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    return(loss, optimizer)

def trainNet(net, batch_size, n_epochs, learning_rate,train_set,val_loader):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    net = net.float().cuda()
    #Get training data
    train_loader = get_train_loader(batch_size,train_set)
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
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        
        for i, data in enumerate(train_loader, 0):
            #print(i)
            #Get inputs
            inputs = data.clone().detach().requires_grad_(True)   
            inputs = inputs.float().cuda()      
            #print(inputs.shape)
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs, mu, logvar = net(inputs)
            outputs = outputs.float().cuda()
            mu = mu.float().cuda()
            logvar = logvar.float().cuda()
            #print(outputs.shape)
            loss_size = loss_fn(outputs, inputs,mu,logvar) # labels)
            optimizer.zero_grad()
            loss_size.backward() #.mean().backward()
            optimizer.step()
            #return outputs
            #plt.imshow(outputs[0,:,:,0],cmap='gray')
            #Print statistics
            running_loss += loss_size.data.item()
            total_train_loss += loss_size.data.item()
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                #print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                print(total_train_loss)
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
            
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inputs in val_loader:

           #Forward pass
            val_outputs,vmu,vlogvar = net(inputs.float().cuda())
            val_loss_size = loss_fn(val_outputs.float().cuda(), inputs.float().cuda(),vmu.float().cuda(),vlogvar.float().cuda()) # labels)
            total_val_loss += val_loss_size.data.item()
        training_losses.append(total_train_loss) 
        validation_losses.append(total_val_loss)
        print(epoch)   
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
    return sample


def doYaThang(path,epochs,load=""):
	transform = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	imagelist = glob.glob(path)
	random.shuffle(imagelist)
	validation_split = 1/3
	validation_amount = int(len(imagelist)*validation_split)
	validation_set = imagelist[:validation_amount]
	training_set = imagelist[validation_amount:]
	whales = WhaleDataset(imagelist=training_set,transform=transform)
	vhales = WhaleDataset(imagelist=validation_set,transform=transform)
	test_loader = torch.utils.data.DataLoader(whales, batch_size=1,  num_workers=2,shuffle=True)
	val_loader = torch.utils.data.DataLoader(vhales, batch_size=1,  num_workers=2,shuffle=True)
	if load:
		CNN = torch.load("cnnsave_" + load)
	else:
		CNN = facenet()
	trainNet(CNN, batch_size=10, n_epochs=epochs, learning_rate=0.001,train_set=whales,val_loader=val_loader)
	torch.save(CNN, str("cnnsave"))


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


def main():
	torch.cuda.set_device(0)
	doYaThang(path="test_images/set/*.jpg",epochs=2000)
	#showOutputs(200)
	

if __name__ == "__main__":
    main()
