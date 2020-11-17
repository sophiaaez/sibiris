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
from skimage import io, util, transform, color
import optuna

#from torch.utils.tensorboard import SummaryWriter


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
    def __init__(self,layer_amount=6,channels=1,isize=512,z_dim=64,layer_size=128):
        super(facenetAE,self).__init__()
        isize=512
        poolamount = 16
        z_dim=z_dim
        bottlefilters = layer_size
        h_dim = int(pow((isize/(poolamount*2)),2)*bottlefilters)
        self.layer_amount = layer_amount
        self.layer_size=layer_size

        #Encoder
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3)  
        self.conv2 = nn.Conv2d(64, 64, 1)
        self.conv2a = nn.Conv2d(64, 192,3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(192, 192,1,stride=1)
        self.conv3a = nn.Conv2d(192,384,3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(384,384,1,stride=1)
        self.conv4a = nn.Conv2d(384,256,3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(256,256,1,stride=1)
        self.conv5a = nn.Conv2d(256,256,3,stride=1,padding=1)
        self.conv6 = nn.Conv2d(256,256,1,stride=1)
        self.conv6a = nn.Conv2d(256,256,3,stride=1,padding=1)

        self.conv41 = nn.Conv2d(256,128,1,stride=1)
        self.conv41a = nn.Conv2d(128,128,3,stride=1,padding=1)
        self.conv42 = nn.Conv2d(256,64,1,stride=1)
        self.conv42a = nn.Conv2d(64,64,3,stride=1,padding=1)
        self.conv43 = nn.Conv2d(256,32,1,stride=1)
        self.conv43a = nn.Conv2d(32,32,3,stride=1,padding=1)

        self.pool = nn.MaxPool2d(3, 2,padding=1)
        self.lrn = nn.LocalResponseNorm(2)
       
        #Bottleneck
        self.fl = Flatten()
        self.unfl = UnFlatten(bottlefilters)

        #Decoder
        self.t_conv41 = nn.ConvTranspose2d(128,256,1,stride=1)
        self.t_conv42 = nn.ConvTranspose2d(64,256,1,stride=1)
        self.t_conv43 = nn.ConvTranspose2d(32,256,1,stride=1)

        self.t_conv4 = nn.ConvTranspose2d(256,384,2,stride=2)
        self.t_conv3 = nn.ConvTranspose2d(384,192,2,stride=2)
        self.t_conv2 = nn.ConvTranspose2d(192,64,2,stride=2)
        self.t_conv1 = nn.ConvTranspose2d(64, 1, 4, stride=4)

        #siamese
        self.l1 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
        self.l2 = nn.Conv2d(64, 64, 3, stride=2,padding=2)
        self.l3 = nn.Conv2d(64, 64, 3, stride=2,padding=2)
        self.b1d32 = nn.BatchNorm1d(32)
        self.bn = nn.BatchNorm2d(64)
        self.bnn = nn.BatchNorm2d(128)
        self.bn1d = nn.BatchNorm1d(128)

        self.l11 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
        self.l22 = nn.Conv2d(64, 64, 3, stride=2,padding=2)
        self.l33 = nn.Conv2d(64, 64, 3, stride=2,padding=2) 

        self.l111 = nn.Conv2d(64, 128, 3, stride=1,padding=1)
        self.l222 = nn.Conv2d(128, 128, 3, stride=2,padding=2)
        self.l333 = nn.Conv2d(64, 128, 3, stride=2,padding=2) 

        self.o = nn.Conv2d(128,256,1,stride=1,padding=1)
        self.fl = Flatten()
        self.d = nn.Linear(9216,512)

        self.c1 = nn.Linear(1024,128)
        self.c2 = nn.Linear(128,32)
        self.c3 = nn.Linear(32,1)

        torch.nn.init.xavier_uniform_(self.conv1.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv2.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv2a.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv3.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv3a.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv4.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv4a.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv5.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv5a.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv6.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv6a.weight,gain=nn.init.calculate_gain('relu'))

        torch.nn.init.xavier_uniform_(self.conv41.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv41a.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv42.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv42a.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv43.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv43a.weight,gain=nn.init.calculate_gain('relu'))


        torch.nn.init.xavier_uniform_(self.t_conv41.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.t_conv42.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.t_conv43.weight,gain=nn.init.calculate_gain('relu'))

        torch.nn.init.xavier_uniform_(self.t_conv4.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.t_conv3.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.t_conv2.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.t_conv1.weight,gain=nn.init.calculate_gain('relu'))




    def forward(self, x):
        #print(x.size())
        x = nn.ReLU()(self.conv1(x)) 
        #print(x.size())
        x = self.pool(x)
        x = self.lrn(x)
        #print(x.size()) #[64, 128, 128]
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv2a(x))
        #print(x.size())
        x = self.lrn(x)
        x = self.pool(x)
        #print(x.size()) #[192, 64, 64]
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv3a(x))
        #print(x.size())
        x = self.pool(x)
        #print(x.size()) #[384,32,32]
        x = nn.ReLU()(self.conv4(x))
        x = nn.ReLU()(self.conv4a(x))
        #print(x.size()) #[256,32,32]
        if(self.layer_amount > 4):
            x = nn.ReLU()(self.conv5(x))
            x = nn.ReLU()(self.conv5a(x))
            #print(x.size())
            if(self.layer_amount > 5):
                    x = nn.ReLU()(self.conv6(x))
                    x = nn.ReLU()(self.conv6a(x))
                    #print(x.size())
        if(self.layer_size == 128):
            x = nn.ReLU()(self.conv41(x))
            x = nn.ReLU()(self.conv41a(x))
            #[128,32,32]
        elif(self.layer_size == 64):
            x = nn.ReLU()(self.conv42(x))
            x = nn.ReLU()(self.conv42a(x))
            #[64,32,32]
        elif(self.layer_size == 32):
            x = nn.ReLU()(self.conv43(x))
            x = nn.ReLU()(self.conv43a(x))
            #[32,32,32]
        #print(x.size()) #[layer_size,32,32]
        x = self.pool(x)
        #print(x.size()) #[layer_size,16,16]
        h=self.fl(x)
        #print(h.size())
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
        x = nn.ReLU()(self.t_conv1(x))
        #print(x.size()) #[1,512,512]
        x = nn.Sigmoid()(x)
        #print(x.size())
              
        return x

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
        return(h)

    def forward_siamese(self,x):
        #print(x.size())
        x = self.unfl(x)
        #print(x.size())
        x1 = nn.ReLU()(self.l1(x))
        x2 = nn.ReLU()(self.l2(x1))
        x3 = self.l3(x)
        x = nn.ReLU()(x2 + x3)
        x = self.bn(x)

        x1 = nn.ReLU()(self.l11(x))
        x2 = self.l22(x1)
        x3 = self.l33(x)
        x = nn.ReLU()(x2 + x3)
        x = self.bn(x)
    
        x1 = nn.ReLU()(self.l111(x))
        x2 = self.l222(x1)
        x3 = self.l333(x)
        x = nn.ReLU()(x2 + x3)
        x = self.bnn(x)

        x = self.o(x)
        x = self.fl(x)
        #print(x.size())
        x = nn.ReLU()(self.d(x))
        
        return x


    def siamese(self,x1,x2):
        o1 = self.encode(x1)
        o2 = self.encode(x2)
        x = self.siamese_only(o1,o2)
        return x

    def siamese_only(self,o1,o2):
        o1 = self.forward_siamese(o1)
        o2 = self.forward_siamese(o2)

        x = torch.cat((o1,o2),axis=-1)
        x = nn.ReLU()(self.c1(x))
        x = self.bn1d(x)
        x = nn.Dropout2d(0.25)(x)
        x = nn.ReLU()(self.c2(x))
        x = self.b1d32(x)
        x = nn.Sigmoid()(self.c3(x))
        return x


def loss_fn(recon_x, x):   # defining loss function for va-AE (loss= reconstruction loss + KLD (to analyse if we have normal distributon))
    #mse = nn.MSELoss()(recon_x,x)
    bce = F.binary_cross_entropy(recon_x, x)
    return bce
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

class WhaleDataset(Dataset):
  def __init__(self,imagelist,img_size,path="../data/kaggle/"):
    self.transform = transform
    self.imagelist = imagelist
    self.img_size = img_size
    self.path = path

  def __len__(self):
    return(len(self.imagelist))

  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    img_name = self.imagelist[idx][0]
    x1,y1,x2,y2 = self.imagelist[idx][1]
    image = io.imread(str(self.path + img_name),as_gray=True)
    image = image[y1:y2,x1:x2] #CROPPING
    image = transform.resize(image,(self.img_size,self.img_size))
    image = transforms.ToTensor()(image)
    image = image.float().cuda()
    return image

  def getDatasetSize(self):
    return(len(self.imagelist))

  def getImageAndName(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    img_name = self.imagelist[idx][0]    
    x1,y1,x2,y2 = self.imagelist[idx][1]
    image = io.imread(str(self.path + img_name),as_gray=True)
    image = image[y1:y2,x1:x2] #CROPPING
    image = transform.resize(image,(self.img_size,self.img_size))
    image = torch.from_numpy(np.array([[image],[image]]))
    #image = transforms.ToTensor()(image)
    return image,img_name

  def getImageAndAll(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    img_name = self.imagelist[idx][0]    
    x1,y1,x2,y2 = self.imagelist[idx][1]
    image = io.imread(str(self.path + img_name),as_gray=True)
    image = image[y1:y2,x1:x2] #CROPPING
    image = transform.resize(image,(self.img_size,self.img_size))
    image = torch.from_numpy(np.array([[image]]))
    tag = ""
    if len(self.imagelist[idx])>2:
        tag = self.imagelist[idx][2]
    return image,img_name,self.imagelist[idx][1],tag
    
  def getNameForIDX(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    img_name = self.imagelist[idx][0]
    return img_name

  def getIDForIDX(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    img_name = self.imagelist[idx][2]
    return img_name
  
  def getBatch(self,batch_size,idx):
    batch = torch.tensor([])
    for i in range(idx*batch_size,(idx+1)*batch_size):
        if i < len(self.imagelist):
            batch = torch.cat([batch,self.__getitem__(i)])
        else:
            batch = torch.cat([batch,self.__getitem__(np.random.randint(0,len(self.imagelist)))])
    return batch.reshape((batch_size,1,self.img_size,self.img_size))

  def getDoubleBatch(self,batch_size,idx): #creates a double batch of images and labels, 1/3 approx. is positive examples
    batch1 = torch.tensor([])
    batch2 = torch.tensor([])
    siam_out = torch.tensor([])
    matches = 0
    for i in range(idx*batch_size,(idx+1)*batch_size):
        if i < len(self.imagelist):
            batch1 = torch.cat([batch1,self.__getitem__(i)])
        else:
            batch1 = torch.cat([batch1,self.__getitem__(np.random.randint(0,len(self.imagelist)))])
        if matches <= int(batch_size/3) and not (self.imagelist[i][2] == 'new_whale') and not (self.imagelist[i][2] == ""):
            for j in range(len(self.imagelist)):
                if self.imagelist[i][2] == self.imagelist[j][2] and not (i == j) and not (self.imagelist[i][2] == 'new_whale') and not (self.imagelist[i][2] == ""):
                    batch2 = torch.cat([batch2,self.__getitem__(j)])
                    siam_out = torch.cat([siam_out,torch.tensor([1])])
                    matches += 1
                    break
        if not (len(batch1) == len(batch2)):
            j = np.random.randint(0,len(self.imagelist))
            while i == j:
                j = np.random.randint(0,len(self.imagelist))
            batch2 = torch.cat([batch2,self.__getitem__(j)])
            if self.imagelist[i][2] == self.imagelist[j][2] and not (i == j) and not (self.imagelist[i][2] == 'new_whale') and not (self.imagelist[i][2] == ""):
                siam_out = torch.cat([siam_out,torch.tensor([0])])
                matches += 1
            else:
                siam_out = torch.cat([siam_out,torch.tensor([0])])
    batch1 = batch1.reshape((batch_size,1,self.img_size,self.img_size))
    batch2 = batch2.reshape((batch_size,1,self.img_size,self.img_size))
    siam_out = siam_out.reshape((batch_size,1))
    return(batch1.float().cuda(),batch2.float().cuda(),siam_out.float().cuda())



def getDatasets(filepath,validation_split=1/3,reduction=0.5):
    size = 512
    set_raw = []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
                name = str(row[0])
                box = (str(row[1])[1:-1]).split(",")
                bbox = [int(b) for b in box]
                if len(row) == 2:
                    label = ""
                elif len(row) == 3:
                    label = str(row[2])
                set_raw.append([name,bbox,label])
    random.shuffle(set_raw)
    set_raw_len  = int(len(set_raw)*reduction)
    validation_amount = int(set_raw_len*validation_split)
    validation_set = set_raw[:validation_amount]
    training_set = set_raw[validation_amount:set_raw_len]
    whales = WhaleDataset(imagelist=training_set,img_size=size)
    vhales = WhaleDataset(imagelist=validation_set,img_size=size)
    #train_loader = torch.utils.data.DataLoader(whales, batch_size=batch_size,  num_workers=2,shuffle=True)
    #val_loader = torch.utils.data.DataLoader(vhales, batch_size=batch_size,  num_workers=2,shuffle=True)
    return whales,vhales #train_loader, val_loader


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


def trainNet(epochs,learning_rate,batch_size,data_path,layers,layer_size,save=True):
    print("AND GO")
    #writer = SummaryWriter('whales')
    #DATA
    train_set,val_set = getDatasets(data_path) #../data/train/crops/",batch_size)
    #MODEL
    model = facenetAE(layer_amount=layers,layer_size=layer_size).cuda()
    #EARLY STOPPER
    es = EarlyStopper(5,0.1,str("AE_earlystopsave_4.pth"),save)
    #writer.add_graph(model,images)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss()
    model.train()
    #AUTOENCODER TRAINING
    training_losses = []
    validation_losses = []
    siamese_losses = []
    for epoch in range(epochs):
        total_train_loss = 0
        total_siamese_loss = 0
        for b in range(int(train_set.getDatasetSize()/batch_size)):
            #inputs = train_set.getBatch(batch_size,b)
            optimizer.zero_grad()
            #inputs = torch.from_numpy(np.array(inputs)).float().cuda()
            b1,b2,l = train_set.getDoubleBatch(batch_size,b)
            outputs = model(b1.float().cuda())
            BCE = loss_fn(outputs, b1)
            o = model.siamese(b1,b2)
            s_loss = mse(o,l)
            loss = s_loss + BCE
            loss.backward() 
            optimizer.step()
            total_train_loss += BCE
            total_siamese_loss += s_loss
        #print("train loss " + str(total_train_loss.detach().cpu().item()))
        """if epoch % 5 == 0: 
            #SIAMESE TRAINING
            siamese_loss = 0
            for b in range(int(train_set.getDatasetSize()/batch_size)):
                optimizer.zero_grad()
                b1,b2,l = train_set.getDoubleBatch(batch_size,b)
                o = model.siamese(b1,b2)
                loss = mse(o,l)
                loss.backward()
                optimizer.step()
                siamese_loss += loss
            siamese_losses.append(siamese_loss.detach().cpu().item())
            print("siamese loss "  + str(siamese_loss.detach().cpu().item()))"""
        #VALIDATION
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                total_val_loss = 0
                for b in range(int(val_set.getDatasetSize()/batch_size)):
                    vinputs = val_set.getBatch(batch_size,b)
                    val_outputs = model(vinputs)
                    vBCE  = loss_fn(val_outputs,vinputs)
                    total_val_loss += vBCE
                    #image  =val_outputs[0,0].cpu().detach()
                    #io.imsave("test_images/set/outputs/output_" + str(epoch) + ".jpg", (color.grey2rgb(image)*255).astype(np.uint8))
            validation_losses.append(total_val_loss.detach().cpu().item())
            print("Epoch " + str(epoch) + " with val loss " + str(validation_losses[-1]))
            model.train() #reset to train
            stop = es.earlyStopping(total_val_loss,model)
            #EARLY STOPPING
            if stop:
                print("TRAINING FINISHED AFTER " + str(epoch) + " EPOCHS. K BYE.")
                break
        siamese_losses.append(total_siamese_loss.detach().cpu().item())
        training_losses.append(total_train_loss.detach().cpu().item()) 
        stop_epoch = epoch
    #writer.close()
    #SAVE LOSSES TO FILEs
    if save:
        filename = str("AE_losses_" + str(layers)+ "_" + str(layer_size) +"_mse_loss.txt")
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
        file.close()

def getAndSaveOutputs(filepath,network_path=None,amount=100):
    imagelist = []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            name = str(row[0])
            box = (str(row[1])[1:-1]).split(",")
            bbox = [int(b) for b in box]
            imagelist.append([name,bbox])
    dataset = WhaleDataset(imagelist,512)
    encoding_ids = None
    encodings = np.array([])
    if network_path:
        model = torch.load(network_path)
    else:
        model = facenetAE()
    if amount > len(dataset):
        amount = len(dataset)
    for i in range(amount):
        img, img_name = dataset.getImageAndName(i)
        output = model.forward(img.float().cuda())
        imagename = img_name.split("/")[-1]
        image  =output[0,0].cpu().detach()
        io.imsave("./trial_run/output_ae/" + imagename, (color.grey2rgb(image)*255).astype(np.uint8))
        print("./trial_run/output_ae/" + imagename)

def getAndSaveEncodings(filepath,network_path=None):
    imagelist = []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            name = str(row[0])
            box = (str(row[1])[1:-1]).split(",")
            bbox = [int(b) for b in box]
            tag = ""
            if(len(row)>2):
                tag = str(row[2])
            imagelist.append([name,bbox,tag])
    dataset = WhaleDataset(imagelist,512)
    #encoding_ids = np.array([])
    encodings = np.array([])
    if network_path:
        model = torch.load(network_path)
    else:
        model = facenetAE()
    for i in range(len(dataset)):
        img, img_name, bbox, tag = dataset.getImageAndAll(i)
        encoding = model.encode(img.float().cuda())
        eco=encoding.detach().cpu().numpy()
        if i == 0:
            encodings = np.array(eco)
            encoding_ids = [[img_name,str(bbox),tag]]#np.array([img_name,bbox,tag])
        else:
            encodings = np.append(encodings,eco,axis=0)
            encoding_ids.append([img_name,str(bbox),tag])# = np.append(encoding_ids,[img_name,str(bbox),tag],axis=0)
        #if i%1000 == 0:
            #print(i)
    e_ids = np.array(encoding_ids)
    with open('ae_training_encodings.npy', 'wb') as f:
        np.save(f, encodings)
    with open('ae_training_encoding_ids.npy','wb') as f:
        np.save(f,encoding_ids)
    #return encodings

def evalSet(filepath,network_path=None):
    #get data
    imagelist = []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            name = str(row[0])
            box = (str(row[1])[1:-1]).split(",")
            bbox = [int(b) for b in box]
            imagelist.append([name,bbox])
    dataset = WhaleDataset(imagelist,512)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2,num_workers=2)
    #get model
    if network_path:
        model = torch.load(network_path)
    else:
        model = facenetAE()
    #calculate loss
    total_loss = 0
    for inputs in loader:
        inputs = inputs.float().cuda()
        outputs = model(inputs)
        BCE  = loss_fn(outputs,inputs)
        total_loss += BCE.detach().cpu().item()
    print("Total loss for testset is: " + str(total_loss))

def objective(trial):
    epochs = 1000
    #learning_rate =trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    #batch_size = trial.suggest_int("batch_size",8,32,8)
    learning_rate = 0.0001
    batch_size = 8
    layer_amount = 4#trial.suggest_int("layer_amount",4,6,1)
    layer_size = trial.suggest_categorical("layer_size",[32,64,128,256])
    #print("BATCH SIZE: " + str(batch_size))
    print("Layer amount: " + str(layer_amount))
    print("Layer size: " + str(layer_size))
    #DATA
    train_loader,val_loader = getDatasets("../data/trainingset_final.csv",batch_size,1/3,reduction=0.25)
    #MODEL
    model = facenetAE(layer_amount=layer_amount,layer_size=layer_size).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #loss = nn.BCEWithLogitsLoss()
    model.train()
    #TRAINING
    training_losses = []
    validation_losses = []
    es = EarlyStopper(10,0.1,str("AE_earlystopsave_" + str(layer_amount) + "_" + str(layer_size)),False)
    for epoch in range(epochs):
        total_train_loss = 0
        model.train()
        for inputs in train_loader:
            optimizer.zero_grad()
            inputs = inputs.float().cuda()
            outputs = model(inputs)
            BCE= loss_fn(outputs, inputs)
            #bcel = loss(outputs.float(),inputs.float())
            BCE.backward() 
            optimizer.step()
            total_train_loss += BCE
        #print("train loss " + str(total_train_loss.detach().cpu().item()))
        #VALIDATION 
        if epoch % 10 == 0: 
            model.eval()
            with torch.no_grad():
                total_val_loss = 0
                for vinputs in val_loader:
                    vinputs = vinputs.float().cuda()
                    val_outputs = model(vinputs)
                    vBCE  = loss_fn(val_outputs, vinputs)
                    total_val_loss += vBCE
            validation_losses.append(total_val_loss.detach().cpu().item())
            print(str(epoch) + "  " + str(total_val_loss.detach().cpu().item()))
            stop = es.earlyStopping(total_val_loss,model)
            #EARLY STOPPING
            if stop:
                print("TRAINING FINISHED AFTER " + str(epoch) + " EPOCHS. K BYE.")
                break
        training_losses.append(total_train_loss.detach().cpu().item()) 
        trial.report(total_val_loss,epoch)
    #if not final_loss:
    final_loss = validation_losses[-1]
    #WRITE OPTIM 
    filename = str("optim.txt")
    file=open(filename,'a')
    file.write("layer_amount:" + str(layer_amount))
    file.write("layer_size:" + str(layer_size))
    file.write("final_loss:" + str(final_loss))  
    file.close()
    return final_loss


def main():
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

def matchTop10():
    model = torch.load("AE_earlystopsave_4.pth").cuda()
    tr_enc = np.load("../ae/ae_training_encodings.npy")
    tr_ids = np.load("../ae/ae_training_encoding_ids.npy")
    te_enc = np.load("../ae/ae_test_encodings.npy")
    te_ids = np.load("../ae/ae_test_encoding_ids.npy")
    tr_idx = findIndices(tr_ids)
    tr_enc = tr_enc[tr_idx]
    tr_ids = tr_ids[tr_idx]
    te_idx = findIndices(te_ids)
    te_enc = te_enc[te_idx]
    te_ids = te_ids[te_idx]
    pred = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(te_ids)):
      te = torch.from_numpy(np.array([te_enc[i],te_enc[i],te_enc[i],te_enc[i],te_enc[i]])).float().cuda()
      matches = []
      for j in range(0,len(tr_ids)-5,5):
        tr = torch.from_numpy(np.array([tr_enc[j],tr_enc[j+1],tr_enc[j+2],tr_enc[j+3],tr_enc[j+4]])).float().cuda()
        res = model.siamese_only(te,tr)
        for r in res:
          if r > 0.5: #classified match 
            matches.append([r.item(),tr_ids[j,-1]])
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
      matches = matches[matches[:,0].argsort()]
      match = 0
      for k in range(1,11):
        m_id = matches[-k,-1]
        if te_ids[i,-1] == m_id:
          match = k
          break
      if match > 0:
        pred.append(1)
      else: 
        pred.append(0)
      #if tp+fp > 0 and tp+fn > 0:
        #print("recall: " + str(tp/(tp+fn)) + " and precision: " + str(tp/(tp+fp)))
    a = np.mean(np.array(pred))
    print("test accuracy: " + str(a))
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    print("recall: " + str(recall) + " and precision: " + str(precision))
    filename = str("ae_top10.txt")
    file=open(filename,'a')
    file.write("recall:" + str(recall))
    file.write("precision :" + str(precision))
    file.write("accuracy:" + str(a)) 
    file.write("matches:" + str(tp))
    file.close()

def main2():
    getAndSaveEncodings("../data/train/crops/")

def main3():
    trainNet(epochs=1000,learning_rate=0.0001,batch_size=8,data_path="../data/trainingset_final.csv",layers=4,layer_size=64,save=True)

def main4():
	getAndSaveOutputs("../data/trainingset_final.csv","AE_earlystopsave_4.pth",amount=10)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main4()
    #main4()
    #getAndSaveEncodings("../data/trainingset_final.csv","AE_earlystopsave_4")
    #evalSet("../data/testset_final.csv","AE_earlystopsave_4")
    #trainNet(epochs=1,learning_rate=0.0001,batch_size=8,data_path="../data/trainingset_final.csv",layers=4)
    
