import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, util, transform, color,exposure,filters
import csv
import numpy as np
import random
from torchvision import transforms

class WhaleDataset(Dataset):
  def __init__(self,imagelist,img_size,path="../data/kaggle/",transformation_share=0.5):
    self.imagelist = imagelist
    self.img_size = img_size
    self.path = path
    #self.print = 0
    """self.transform = nn.Sequential(
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(45), #doesnt work
            #transforms.ColorJitter(),
            transforms.RandomAffine(20,fillcolor=125),
            transforms.GaussianBlur(kernel_size=5),
            #transforms.Grayscale()
    )
    self.scripted_transforms = torch.jit.script(self.transform)"""
    self.ts = transformation_share

  def __len__(self):
    return(len(self.imagelist))

  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    img_name = self.imagelist[idx][0]
    x1,y1,x2,y2 = self.imagelist[idx][1]
    #image = Image.open(str(self.path + img_name)).convert("L")
    #image = image.crop((x1,y1,x2,y2))
    #image = image.resize((self.img_size,self.img_size))
    #image = transforms.ToTensor()(image)
    image = io.imread(str(self.path + img_name),as_gray=True)
    image = image[y1:y2,x1:x2] #CROPPING
    image = transform.resize(image,(self.img_size,self.img_size))
    tran = np.random.randint(0,100)
    #image = transforms.ToTensor()(image)
    #image = self.scripted_transforms(image)
    if tran < self.ts*100:
        tran_1 = np.random.randint(0,2)
        if tran_1 == 1: #rescale intensity
            low = np.random.randint(0,21)*0.01
            high = np.random.randint(80,101)*0.01
            image = exposure.rescale_intensity(image,in_range=(low,high),out_range=(0.0,1.0))
        tran_1 = np.random.randint(0,2)
        if tran_1 == 1: #add noise
            image = util.random_noise(image)
        tran_1 = np.random.randint(0,2)
        if tran_1 ==  1: #rotation
            rotate = np.random.randint(-20,21)
            image = transform.rotate(image,rotate,mode='edge')
        tran_1 = np.random.randint(0,2)
        if tran_1 ==  1: #gaussian blur
            sig = np.random.randint(0,21)*0.1
            image = filters.gaussian(image,sigma=sig)
        tran_1 = np.random.randint(0,2)
        if tran_1 ==  1: #flip horizontally
            image = image[:, ::-1]
            image = image.copy()
    image = transforms.ToTensor()(image)  
    return image

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

def getDatasets(filepath,batch_size,validation_split=1/3,reduction=0.25):
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
    train_loader = torch.utils.data.DataLoader(whales, batch_size=batch_size,  num_workers=2,shuffle=True)
    val_loader = torch.utils.data.DataLoader(vhales, batch_size=batch_size,  num_workers=2,shuffle=True)
    return train_loader, val_loader