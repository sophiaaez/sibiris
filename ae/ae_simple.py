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
    def __init__(self,layer_amount=6,channels=1,isize=512,z_dim=64):
        super(facenetAE,self).__init__()
        isize=512
        poolamount = 16
        z_dim=z_dim
        bottlefilters = 256
        h_dim = int(pow((isize/(poolamount*2)),2)*bottlefilters)
        self.layer_amount = layer_amount

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
        self.pool = nn.MaxPool2d(3, 2,padding=1)
       
        #Bottleneck
        self.fl = Flatten()
        self.unfl = UnFlatten(bottlefilters)

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
        torch.nn.init.xavier_uniform_(self.conv5.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv5a.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv6.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv6a.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.t_conv1.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.t_conv2.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.t_conv3.weight,gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.t_conv4.weight,gain=nn.init.calculate_gain('relu'))



    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.pool(x)
        #print(x.size())
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv2a(x))
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
        x = self.pool(x)
        #print(x.size())
        h=self.fl(x)
        x=self.unfl(h)
        x = nn.ReLU()(self.t_conv4(x))
        x = nn.ReLU()(self.t_conv3(x))
        #print(x.size())
        x = nn.ReLU()(self.t_conv2(x))
        x = nn.ReLU()(self.t_conv1(x))
        #print(x.size())
        x = nn.Sigmoid()(x)
        #print(x.size())
              
        return x

    def encode(self,x):
        x = nn.ReLU()(self.conv1(x))
        x = self.pool(x)
        #print(x.size())
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = self.pool(x)
        #print(x.size())
        x = nn.ReLU()(self.conv4(x))
        x = nn.ReLU()(self.conv5(x))
        x = self.pool(x)
        #print(x.size())
        x = nn.ReLU()(self.conv6(x))
        x = nn.ReLU()(self.conv7(x))
        if(self.layer_amount > 4):
            x = nn.ReLU()(self.conv5(x))
            x = nn.ReLU()(self.conv5a(x))
            if(self.layer_amount > 5):
                    x = nn.ReLU()(self.conv6(x))
                    x = nn.ReLU()(self.conv6a(x))
        x = self.pool(x)
        h=self.fl(x)
        return h

def loss_fn(recon_x, x):   # defining loss function for va-AE (loss= reconstruction loss + KLD (to analyse if we have normal distributon))
    a = round(torch.min(recon_x).item(),5)
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
    train_loader,val_loader = getDatasets("../data/train/crops/",batch_size) #../data/train/crops/",batch_size)
    #dataiter = iter(train_loader)
    #images = dataiter.next()
    #MODEL
    model = facenetAE(layer_amount=6).cuda()
    #writer.add_graph(model,images)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #loss = nn.BCEWithLogitsLoss()
    model.train()
    #TRAINING
    training_losses = []
    validation_losses = []
    for epoch in range(epochs):
        total_train_loss = 0
        for inputs in train_loader:
            optimizer.zero_grad()
            inputs = inputs.float().cuda()
            outputs = model(inputs)
            BCE = loss_fn(outputs, inputs)
            #bcel = loss(outputs.float(),inputs.float())
            BCE.backward() 
            optimizer.step()
            total_train_loss += BCE
        print("train loss " + str(total_train_loss.detach().cpu().item()))
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
    #writer.close()
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
        output,mu,logvar = model.forward(img.float().cuda())
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
    #learning_rate =trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    #batch_size = trial.suggest_int("batch_size",8,32,8)
    learning_rate = 0.001
    batch_size = 16
    layer_amount = trial.suggest_int("layer_amount",4,6,1)
    #print("BATCH SIZE: " + str(batch_size))
    print("Layer amount: " + str(layer_amount))
    #DATA
    train_loader,val_loader = getDatasets("../data/train/crops/",batch_size)
    #MODEL
    model = facenet(layer_amount).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #loss = nn.BCEWithLogitsLoss()
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
                for vinputs in val_loader:
                    vinputs = vinputs.float().cuda()
                    val_outputs,vmu,vlogvar = model(vinputs)
                    vBCEKLD, vBCE, vKLD  = loss_fn(val_outputs, vinputs,vmu,vlogvar)
                    total_val_loss += vBCEKLD
                    image  =val_outputs[0,0].cpu().detach()
        training_losses.append(total_train_loss.detach().cpu().item()) 
        trial.report(total_train_loss,epoch)
    #if not final_loss:
    final_loss = training_losses[-1]
    #WRITE OPTIM 
    filename = str("optim.txt")
    file=open(filename,'w')
    file.write("layer_amount:" + str(layer_amount))
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
    trainNet(epochs=100,learning_rate=0.001,batch_size=32,save=True,data_path="test_images/set/")

def main4():
	getAndSaveOutputs("test_images/","modelsave_t_230e")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main3()
    
