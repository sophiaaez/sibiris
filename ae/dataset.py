import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, util, transform, color,exposure,filters
import csv
import numpy as np
import random
from torchvision import transforms

class WhaleDataset(Dataset):
    def __init__(self,imagelist,img_size,path="../data/kaggle/",transformation_share=0.5,augment=True):
        self.imagelist = imagelist
        self.img_size = img_size
        self.path = path
        self.labelledDoublesImagelist = self.findLabelledDoubles()
        self.augment = augment
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

    def __getitem__(self,idx,fulllist=True):
        if torch.is_tensor(idx):
          idx = idx.tolist()
        if fulllist:
            img_name = self.imagelist[idx][0]
            x1,y1,x2,y2 = self.imagelist[idx][1]
        else:
            img_name = self.labelledDoublesImagelist[idx][0]
            x1,y1,x2,y2 = self.labelledDoublesImagelist[idx][1]
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
        if self.augment and tran < self.ts*100:
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

    def findMatchToX(self,x): #x is the index of the image at the labelled list we're looking for an image
        start = np.random.randint(0,len(self.labelledDoublesImagelist))#start at random point within list
        for i in range(start,len(self.labelledDoublesImagelist)): #search until end of list
            if not (i==x) and self.labelledDoublesImagelist[x][2] == self.labelledDoublesImagelist[i][2]:
                return i
        for i in range(0,start+1): #search until starting point
            if not (i==x) and self.labelledDoublesImagelist[x][2] == self.labelledDoublesImagelist[i][2]:
                return i
        return(-1) #return failure

    def getDoubleBatch(self,batch_size, idx,everyX=2):
        batch1 = torch.tensor([])
        batch2 = torch.tensor([])
        siam_out = torch.tensor([])
        matches = 0
        for i in range(idx*batch_size,(idx+1)*batch_size):
            if i >= len(self.labelledDoublesImagelist):
                i = np.random.randint(0,len(self.labelledDoublesImagelist))
            batch1 = torch.cat([batch1,self.__getitem__(i,False).cuda()])
            #if we've not fulfilled the everyX quota
            if matches < int(batch_size/everyX):
                idx = self.findMatchToX(i)
                if idx > -1:
                        batch2 = torch.cat([batch2,self.__getitem__(idx,False).cuda()])
                        siam_out = torch.cat([siam_out,torch.tensor([0]).cuda()])#match
                        matches += 1
            else:
                j = np.random.randint(0,len(self.labelledDoublesImagelist)) #find random one
                while self.labelledDoublesImagelist[i][2] == self.labelledDoublesImagelist[j][2]: #check if it's the same label
                    j = np.random.randint(0,len(self.labelledDoublesImagelist)) #and if so draw again until it's not
                batch2 = torch.cat([batch2,self.__getitem__(j,False).cuda()]) 
                siam_out = torch.cat([siam_out,torch.tensor([1]).cuda()]) #no match

            """#if no match was found/the batch1 is still shorter than batch2
            if not (len(batch1) == len(batch2)):
                j = np.random.randint(0,len(self.labelledDoublesImagelist)) #find random one
                while i == j: #check if it's not the same we got
                    j = np.random.randint(0,len(self.labelledDoublesImagelist)) #or draw again
                batch2 = torch.cat([batch2,self.__getitem__(j).cuda()]) 
                if self.labelledDoublesImagelist[i][2] == self.labelledDoublesImagelist[j][2] and not (i == j): #just to be save
                    siam_out = torch.cat([siam_out,torch.tensor([0]).cuda()])#match
                    matches += 1
                else:
                    siam_out = torch.cat([siam_out,torch.tensor([1]).cuda()]) #no match"""
        batch1 = batch1.reshape((batch_size,1,self.img_size,self.img_size))
        batch2 = batch2.reshape((batch_size,1,self.img_size,self.img_size))
        siam_out = siam_out.reshape((batch_size,1))
        return(batch1.float().cuda(),batch2.float().cuda(),siam_out.float().cuda())

    def getDatasetSize(self,onlyLabelledAndDouble=False):
        if onlyLabelledAndDouble:
            return len(self.labelledDoublesImagelist)
        else:
            return len(self.imagelist)

    def findLabelledDoubles(self):
        size = 0
        labelled = []
        for i in self.imagelist:
            if len(i) > 2 and not(i[2] == "new_whale" or i[2] == ""):
                labelled.append(i)
        unique = {}
        uniqueDoubles = set()
        for l in labelled:
            if len(l)>2 and (not l[2] in unique.keys()):
                unique[l[2]] = 1
            elif len(l)>2 and l[2] in unique.keys():
                unique[l[2]] = unique[l[2]] + 1
                uniqueDoubles.add(l[2])
        labelledDoubles = []
        ilist = np.array(self.imagelist)
        for u in uniqueDoubles:
            idx = np.where(ilist[:,2] == u)
            labelledDoubles.extend(ilist[idx[0]])
        return labelledDoubles




def getDatasets(filepath,batch_size,validation_split=1/3,reduction=1,raw=False,augment=True):
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
    if len(validation_set) > 0:
        vhales = WhaleDataset(imagelist=validation_set,img_size=size)
        val_loader = torch.utils.data.DataLoader(vhales, batch_size=batch_size,  num_workers=2,shuffle=True)
    else:
        vhales = 0
        val_loader = 0
    train_loader = torch.utils.data.DataLoader(whales, batch_size=batch_size,  num_workers=2,shuffle=True)
    if raw:
        return(train_loader,val_loader,whales,vhales)
    else:
        return(train_loader, val_loader)