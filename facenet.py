import torch
import torchvision
from torch import nn
from torch.autograd import Variable
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


class facenet(nn.Module):
  def __init__(self):
    super(facenet,self).__init__()
    self.encoder = nn.Sequential( #input 1 500 1000
            #Layer1
            nn.Conv2d(1, 64, 7, stride=2, padding=3),  #64, 250, 500
            nn.ReLU(True),
            nn.MaxPool2d(3,stride=2,padding=1),  #64, 125, 250

            #nn.LocalResponse(2),
            #Layer2
            nn.Conv2d(64, 64,1,stride=1), #64, 125, 250
            nn.ReLU(True),
            nn.Conv2d(64, 192,3,stride=1,padding=1), #192, 125, 250
            nn.ReLU(True), 
            #nn.LocalResponse(2),
            nn.MaxPool2d(3,stride=2,padding=1), #192, 63, 125
            #Layer3
            nn.Conv2d(192, 192,1,stride=1), #192, 63, 125
            nn.ReLU(True),
            nn.Conv2d(192,384,3,stride=1,padding=1), #384, 63, 125
            nn.ReLU(True), 
            nn.MaxPool2d(3,stride=2,padding=1), #384, 32, 63
            #Layer4
            nn.Conv2d(384,384,1,stride=1), #384, 32, 63
            nn.ReLU(True),
            nn.Conv2d(384,256,3,stride=1,padding=1), #256, 32, 63
            nn.ReLU(True),
            #Layer5
            nn.Conv2d(256,256,1,stride=1), #256, 32, 63
            nn.ReLU(True),
            nn.Conv2d(256,256,3,stride=1,padding=1), #256, 32, 63
            nn.ReLU(True),
            #Layer6
            nn.Conv2d(256,256,1,stride=1), #256, 32, 63
            nn.ReLU(True),
            nn.Conv2d(256,256,3,stride=1,padding=1), #256, 32, 63
            nn.ReLU(True),
            nn.MaxPool2d(3,stride=2,padding=1), #256, 16, 32

    )

    self.decoder = nn.Sequential(
            #Layer-6
            nn.ConvTranspose2d(256,256,3, stride=(2,2),padding=(1,1)), #256, 31, 63
            nn.ReLU(True),
            nn.ConvTranspose2d(256,256,1, stride=1), #256, 31, 63
            nn.ReLU(True),
            #Layer-5
            nn.ConvTranspose2d(256,256,3, stride=1,padding=(1,1)), #256, 31, 63
            nn.ReLU(True),
            nn.ConvTranspose2d(256,256,1, stride=1), #256, 31, 63
            nn.ReLU(True),
            #Layer-4
            nn.ConvTranspose2d(256,384,3, stride=1,padding=(1,1)), #384, 31, 63
            nn.ReLU(True),
            nn.ConvTranspose2d(384,384,1, stride=1), #384, 31, 63
            nn.ReLU(True),
            #Layer-3
            nn.ConvTranspose2d(384,192,3, stride=(2,2),padding=(0,1)), #192, 63, 125
            nn.ReLU(True),
            nn.ConvTranspose2d(192,192,1, stride=1), #192, 63, 125
            nn.ReLU(True),
            #Layer-2
            nn.ConvTranspose2d(192,64,3, stride=(2,2),padding=(1,1)), #64, 125, 249
            nn.ReLU(True),
            nn.ConvTranspose2d(64,64,1, stride=1), #64, 125, 249
            nn.ReLU(True),
            #Layer-1
            nn.ConvTranspose2d(64,1,7, stride=(4,4),padding=(2,0),output_padding=(1)), #1, 503, 999
            nn.ReLU(True),
            nn.Tanh()
            
    )


  def forward(self,x):
    #print(x.shape)
    x = self.encoder(x)
    #print(x.shape)
    out = self.decoder(x)
    #print(out.shape)
    return(out)


def outputSize(in_size, kernel_size,stride,padding):
  output = int((in_size - kernel_size + 2*(padding))/stride)+1
  return(output)

def get_train_loader(batch_size,train_set):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=2)
    return(train_loader)


def createLossAndOptimizer(net, learning_rate=0.001):
    
    #Loss function
    loss = nn.BCEWithLogitsLoss()
    
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
    net = net.float()
    #Get training data
    train_loader = get_train_loader(batch_size,train_set)
    n_batches = len(train_loader)
    
    #Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)
    
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
            #print(inputs.shape)
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs = net(inputs.float())
            #print(outputs.shape)
            loss_size = loss(outputs.float(), inputs.float()) # labels)
            loss_size.backward()
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
            val_outputs = net(inputs.float())
            val_loss_size = loss(val_outputs.float(), inputs.float()) # labels)
            total_val_loss += val_loss_size.data.item()
        training_losses.append(total_train_loss) 
        validation_losses.append(total_val_loss)   
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
    #Save validation and training losses to file
    filename = str("losses" + str(time.time()) + ".txt")
    file=open(filename,'w')
    file.write("training_losses")
    file.write('\n')
    for element in training_losses:
        file.write(element)
        file.write('\n')
    file.write("validation_losses")
    file.write('\n')
    for element in validation_losses:
        file.write(element)
        file.write('\n')   
    file.close()    
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))



class WhaleDataset(Dataset):
  def __init__(self,imagelist,transform=None):
    self.root_dir = path
    self.transform = transform
    self.imagelist = imagelist

  def __len__(self):
    return(len(self.imagelist))
  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    img_name = self.imagelist[idx]
    sizew = 500
    sizeh = 1000
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


def doYaThang(path,epochs,load=False):
	transform = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    imagelist = glob.glob(path)
    imagelist_s = random.shuffle(imagelist)
    validation_split = 0.33
    validation_amount = int(len(imagelist_s)*validation_split)
    validation_set = imagelist_s[:validation_amount]
    training_set = imagelist_s[validation_amount:]
	whales = WhaleDataset(imagelist=training_set,transform=transform)
	vhales = WhaleDataset(imagelist=validation_set,transform=transform)
	test_loader = torch.utils.data.DataLoader(whales, batch_size=4,  num_workers=2)
	val_loader = torch.utils.data.DataLoader(vhales, batch_size=3,  num_workers=2)
	if load:
		CNN = torch.load(str(path + "cnnsave"))
	else:
		CNN = facenet()
	trainNet(CNN, batch_size=10, n_epochs=epochs, learning_rate=0.001,train_set=whales,val_loader=val_loader)
	torch.save(CNN, str(path + "cnnsave"))



def showOutputs(path):
	CNN.torch.load(str(path + "cnnsave"))
	vhales = WhaleDataset(path=str(path + "test/*.jpg"),transform=transform)
	#val_loader = torch.utils.data.DataLoader(vhales, batch_size=3,  num_workers=2)
	outputs = CNN(vhales)
	return outputs
