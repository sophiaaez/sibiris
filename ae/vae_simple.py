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



class facenet(nn.Module):
    def __init__(self,layer_amount=6,channels=1,isize=512,z_dim=64,layer_size=128):
        super(facenet,self).__init__()
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
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.unfl = UnFlatten(bottlefilters)

        #Decoder
        self.t_conv41 = nn.ConvTranspose2d(128,256,1,stride=1)
        self.t_conv42 = nn.ConvTranspose2d(64,256,1,stride=1)
        self.t_conv43 = nn.ConvTranspose2d(32,256,1,stride=1)

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
        x = nn.ReLU()(self.conv1(x)) 
        x = self.pool(x)
        x = self.lrn(x)
        #print(x.size()) #[64, 128, 128]
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv2a(x))
        x = self.lrn(x)
        x = self.pool(x)
        #print(x.size()) #[192, 64, 64]
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv3a(x))
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
        z,mu,logvar=self.bottleneck(h)
        hz=self.fc3(z)
        x=self.unfl(hz)
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
              
        return x,mu,logvar

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

def init_weights(m): #intialises weights with normal distribution over entire network
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight,gain=nn.init.calculate_gain('relu'))

def loss_fn(recon_x, x, mu, logvar,beta):   # defining loss function for va-AE (loss= reconstruction loss + KLD (to analyse if we have normal distributon))
    a = round(torch.min(recon_x).item(),5)
    b = round(torch.max(recon_x).item(),5)
    c = round(torch.min(x).item(),5)
    d = round(torch.max(x).item(),5)
    #if (a < 0.) or (b > 1.) or (c < 0.) or (d > 1.) or math.isnan(a) or math.isnan(b) or math.isnan(c) or math.isnan(d):
        
    if (a >= 0.) and (b <= 1.) and (c >= 0.) and (d <= 1.) and not (math.isnan(a) or math.isnan(b) or math.isnan(c) or math.isnan(d)):
        BCE = F.binary_cross_entropy(recon_x, x)
        #BCEL = nn.BCEWithLogitsLoss(recon_x, x)
        # source: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # KLD is equal to 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + (beta * KLD), BCE, KLD
    else:
        print("STOPSTOPSTOPSTOPSTOP")
        print("a" + str(a) + "b" + str(b) + "c" + str(c) + "d" + str(d))
        return 0,0,0

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
    return image

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

def getDatasets(filepath,batch_size,validation_split=1/3,reduction=1):
    size = 512
    set_raw = []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
                name = str(row[0])
                box = (str(row[1])[1:-1]).split(",")
                bbox = [int(b) for b in box]
                set_raw.append([name,bbox])
    random.shuffle(set_raw)
    set_raw_len  = int(len(set_raw)*reduction)
    validation_amount = int(set_raw_len*validation_split)
    validation_set = set_raw[:validation_amount]
    training_set = set_raw[validation_amount:set_raw_len]
    whales = WhaleDataset(imagelist=training_set,img_size=size)
    vhales = WhaleDataset(imagelist=validation_set,img_size=size)
    train_loader = torch.utils.data.DataLoader(whales, batch_size=batch_size,  num_workers=2,shuffle=True)
    val_loader = torch.utils.data.DataLoader(vhales, batch_size=batch_size,  num_workers=2,shuffle=True)
    return train_loader, val_loader


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


def getMISTDatasets(batch_size):
    transformer = transforms.Compose([transforms.ToTensor()])
    trainset=datasets.MNIST(root='data', train=True, download=True, transform=transformer)
    testset=datasets.MNIST(root='data', train=False, download=True, transform=transformer)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=0)
    return train_loader, test_loader


def trainNet(epochs,learning_rate,batch_size,data_path,layers,layer_size,beta=1,save=True):
    #writer = SummaryWriter('whales')
    #DATA
    train_loader,val_loader = getDatasets(data_path,batch_size) #../data/train/crops/",batch_size)
    #MODEL
    model = facenet(layer_amount=layers).cuda()
    #EARLY STOPPER
    es = EarlyStopper(10,0.1,str("VAE_earlystopsave_" + str(layers) + "_" + str(layer_size)),save)
    #writer.add_graph(model,images)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    #TRAINING
    training_losses = []
    validation_losses = []
    for epoch in range(epochs):
        total_train_loss = 0
        for inputs in train_loader:
            optimizer.zero_grad()
            inputs = inputs.float().cuda()
            outputs, mu, logvar = model(inputs)
            BCEKLD, BCE, KLD = loss_fn(outputs, inputs,mu,logvar,beta)
            BCEKLD.backward() 
            optimizer.step()
            total_train_loss += BCEKLD
        #print("train loss " + str(total_train_loss.detach().cpu().item()))
        #VALIDATION 
        if epoch % 10 == 0: 
            model.eval()
            with torch.no_grad():
                total_val_loss = 0
                for vinputs in val_loader:
                    vinputs = vinputs.float().cuda()
                    val_outputs,vmu,vlogvar = model(vinputs)
                    vBCEKLD, vBCE, vKLD  = loss_fn(val_outputs, vinputs,vmu,vlogvar,beta)
                    total_val_loss += vBCEKLD
                    #image  =val_outputs[0,0].cpu().detach()
                    #io.imsave("test_images/set/outputs/output_" + str(epoch) + ".jpg", (color.grey2rgb(image)*255).astype(np.uint8))
            validation_losses.append(total_val_loss.detach().cpu().item())
            #print("Epoch " + str(epoch) + " with val loss " + str(validation_losses[-1]))
            model.train() #reset to train
            stop = es.earlyStopping(total_val_loss,model)
            #EARLY STOPPING
            if stop:
                print("TRAINING FINISHED AFTER " + str(epoch) + " EPOCHS. K BYE.")
                stop_epoch = epoch
                break
        training_losses.append(total_train_loss.detach().cpu().item()) 
    #writer.close()
    #SAVE LOSSES TO FILEs
    if save:
        filename = str("VAE_losses_" + str(layers)+ "_"+ str(layer_size)+".txt")
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

def getAndSaveOutputs(filepath,network_path=None):
    #imagelist = glob.glob(image_path + str("*.jpg"))
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
        model = facenet()
    for i in range(len(dataset)):
        img, img_name = dataset.getImageAndName(i)
        output,mu,logvar = model.forward(img.float().cuda())
        imagename = img_name.split("/")[-1]
        image  =output[0,0].cpu().detach()
        io.imsave("./trial_run/output_vae4l/" + imagename, (color.grey2rgb(image)*255).astype(np.uint8))
        print("./trial_run/output_vae4l/" + imagename)

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
        model = facenet()
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
    with open('vae_test_encodings.npy', 'wb') as f:
        np.save(f, encodings)
    with open('vae_test_encoding_ids.npy','wb') as f:
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
        model = facenet()
    #calculate loss
    total_bk = 0
    total_bce = 0
    for inputs in loader:
        inputs = inputs.float().cuda()
        outputs, mu, logvar = model(inputs)
        BCEKLD, BCE, KLD = loss_fn(outputs, inputs,mu,logvar)
        total_bk += BCEKLD.detach().cpu().item()
        total_bce += BCE.detach().cpu().item()
    print("Total loss for testset is: " + str(total_bk))
    print("Total BCE loss for testset is: " + str(total_bce))


def objective(trial):
    epochs = 1000
    #learning_rate =trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    #batch_size = trial.suggest_int("batch_size",8,32,8)
    learning_rate = 0.0001
    batch_size = 8
    layer_amount = 4 #trial.suggest_int("layer_amount",4,6,1)
    layer_size = 128 #trial.suggest_categorical("layer_size",[32,64,128,256])
    beta = trial.suggest_int("beta",1,20,1)
    #print("Layer amount: " + str(layer_amount))
    #print("Layer size: " + str(layer_size))
    print("Beta: " + str(beta))
    #DATA
    train_loader,val_loader = getDatasets("../data/trainingset_final.csv",batch_size,1/3,reduction=0.25)
    #MODEL
    model = facenet(layer_amount=layer_amount,layer_size=layer_size).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #loss = nn.BCEWithLogitsLoss()
    model.train()
    #TRAINING
    training_losses = []
    validation_losses = []
    es = EarlyStopper(10,0.1,str("VAE_earlystopsave_" + str(layer_amount) + "_" + str(layer_size)),False)
    for epoch in range(epochs):
        total_train_loss = 0
        model.train()
        for inputs in train_loader:
            optimizer.zero_grad()
            inputs = inputs.float().cuda()
            outputs, mu, logvar = model(inputs)
            BCEKLD, BCE, KLD = loss_fn(outputs, inputs,mu,logvar,beta)
            BCEKLD.backward() 
            optimizer.step()
            total_train_loss += BCE
        #VALIDATION 
        if epoch % 10 == 0: 
            model.eval()
            with torch.no_grad():
                total_val_loss = 0
                for vinputs in val_loader:
                    vinputs = vinputs.float().cuda()                    
                    voutputs, vmu, vlogvar = model(vinputs)
                    vBCEKLD, vBCE, vKLD = loss_fn(voutputs, vinputs,vmu,vlogvar,beta)
                    total_val_loss += vBCE
            validation_losses.append(total_val_loss.detach().cpu().item())
            print(str(epoch) + "  " + str(total_val_loss.detach().cpu().item()))
            stop = es.earlyStopping(total_val_loss,model)
            #EARLY STOPPING
            if stop or (int(total_val_loss.detach().cpu().item())) == 131:
                print("TRAINING FINISHED AFTER " + str(epoch) + " EPOCHS. K BYE.")
                break
        training_losses.append(total_train_loss.detach().cpu().item()) 
        trial.report(total_val_loss,epoch)
    #if not final_loss:
    final_loss = validation_losses[-1]
    #WRITE OPTIM 
    filename = str("vae_optim.txt")
    file=open(filename,'a')
    file.write("layer_amount:" + str(layer_amount))
    file.write("layer_size:" + str(layer_size))
    file.write("beta:" + str(beta))
    file.write("final_loss:" + str(final_loss))  
    file.close()
    return final_loss


def main():
    torch.cuda.set_device(0)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=10)

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
    trainNet(epochs=20,learning_rate=0.0001,batch_size=8,data_path="../data/trainingset_final.csv",layers=4,save=True)

def main4():

	getAndSaveOutputs("../data/trial_run_test.csv","VAE_earlystopsave_4")

def main5():
    tset, vset = getDatasets("../data/trainingset_final.csv",8)
    tset.getImageAndName(0)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
    #evalSet("../data/testset_final.csv","VAE_earlystopsave_4")
