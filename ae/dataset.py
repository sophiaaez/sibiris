import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from skimage import io, util, transform, color,exposure,filters
import csv
import numpy as np
import random
from torchvision import transforms
import warnings

"""
The Whale Dataset handles the whale images for the autoencoders,  created from the following parameters:
imagelist = a list of images in the format where a row is structured as follows [imagename, bounding box, tag/id]
img_size = the square size which the images of the dataset shall, e.g. 512 for 512x512
path = where the images are located relative to this file, default "../data/kaggle/"
transformation_share = how many of the images are ought to be augmented, default 0.5
augment = if any of the images shall be augmented, default True
findDoubles = if a list of those individuals that appear at least twice shall be kept, default True
"""
class WhaleDataset(Dataset):
    def __init__(self,imagelist,img_size,path="../data/kaggle/",transformation_share=0.5,augment=True,findDoubles=False):
        self.imagelist = imagelist
        self.img_size = img_size
        self.path = path
        self.findDoubles = findDoubles
        if findDoubles:
            self.labelledDoublesImagelist = self.findLabelledDoubles()
        self.augment = augment
        self.transform = nn.Sequential(
                transforms.RandomAffine(20),
        )
        self.scripted_transforms = torch.jit.script(self.transform)
        self.ts = transformation_share
        warnings.simplefilter(action='ignore', category=UserWarning) #coz random affine keeps throwing them!

    """
    Returns the length of the dataset.
    """
    def __len__(self):
        return(len(self.imagelist))

    """
    Returns the item at index idx. fullist determines whether we require images from the labelledDoubles List (False) or not (True), default True.
    Images are cropped according to their bounding box annotation, converted into grayscale and transformed into 512x512 pixels.
    Furthermore, images are augmented, if previously defined, with a chance of 50% for each augmentation: 
    intensity rescaling, additive noise, rotation, gaussian blur, horizontal flip and affine transformation.
    """
    def __getitem__(self,idx,fulllist=True):
        if torch.is_tensor(idx):
          idx = idx.tolist()
        if fulllist:
            img_name = self.imagelist[idx][0]
            x1,y1,x2,y2 = self.imagelist[idx][1]
        elif self.findDoubles:
            img_name = self.labelledDoublesImagelist[idx][0]
            x1,y1,x2,y2 = self.labelledDoublesImagelist[idx][1]
        image = io.imread(str(self.path + img_name),as_gray=True)
        image = image[y1:y2,x1:x2] #CROPPING
        image = transform.resize(image,(self.img_size,self.img_size))
        tran = np.random.randint(0,100)
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
            tran_1 = np.random.randint(0,2)
            if tran_1 ==  1: #affine transform
                image = self.scripted_transforms(image)
        else:    
            image = transforms.ToTensor()(image)  
        return image

    """
    Returns an image, as _getitem_ but the information of the image is added. 
    The output looks as follows: image, image name, bounding box, tag/id.
    """
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

    """
    Finds another image with the same id as that of the image at the location of x.
    The list is searched from a random starting point to the end and from the start to the random starting point.
    Returns the index of this other image or -1 for a failure. 
    """
    def findMatchToX(self,x): #x is the index of the image at the labelled list we're looking for an image
        if self.findDoubles:
            start = np.random.randint(0,len(self.labelledDoublesImagelist))#start at random point within list
            for i in range(start,len(self.labelledDoublesImagelist)): #search until end of list
                if not (i==x) and self.labelledDoublesImagelist[x][2] == self.labelledDoublesImagelist[i][2]:
                    return i
            for i in range(0,start+1): #search until starting point
                if not (i==x) and self.labelledDoublesImagelist[x][2] == self.labelledDoublesImagelist[i][2]:
                    return i
            return(-1) #return failure

    """
    Creates a Double Batch,two batches that align and a label vector that expresses which images on the alignment are from the same individual.
    Given the batch size batch_size and the index idx. This index does not refer to an image, but a batch, therefore idx = 1 -> images from 64-128 if batch_size 64.
    everyX refers to every how many images a match shall be included, default 2.
    Returns one batch, another batch some of whose ids are the same as the first and a label vector that states which of these images match.
    """
    def getDoubleBatch(self,batch_size, idx,everyX=2):
        if self.findDoubles:
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
                    index = self.findMatchToX(i)
                    if index > -1:
                            batch2 = torch.cat([batch2,self.__getitem__(index,False).cuda()])
                            siam_out = torch.cat([siam_out,torch.tensor([0]).cuda()])#match
                            matches += 1
                else:
                    j = np.random.randint(0,len(self.labelledDoublesImagelist)) #find random one
                    while self.labelledDoublesImagelist[i][2] == self.labelledDoublesImagelist[j][2]: #check if it's the same label
                        j = np.random.randint(0,len(self.labelledDoublesImagelist)) #and if so draw again until it's not
                    batch2 = torch.cat([batch2,self.__getitem__(j,False).cuda()]) 
                    siam_out = torch.cat([siam_out,torch.tensor([1]).cuda()]) #no match
            batch1 = batch1.reshape((batch_size,1,self.img_size,self.img_size))
            batch2 = batch2.reshape((batch_size,1,self.img_size,self.img_size))
            siam_out = siam_out.reshape((batch_size,1))
            return(batch1.float().cuda(),batch2.float().cuda(),siam_out.float().cuda())

    """
    Returns the size of the dataset based on onlyLabelledAndDouble (default False). 
    This means that the amount of images that are from a whale that is also portrayed on one or more images can be accessed.
    """
    def getDatasetSize(self,onlyLabelledAndDouble=False):
        if onlyLabelledAndDouble and self.findDoubles:
            return len(self.labelledDoublesImagelist)
        else:
            return len(self.imagelist)

    """
    Creates a list of labelled doubles. Meaning that all those whale ids that appear more than once are searched and all the images
    belonging to these whales are returned in one arraylist.
    """
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

"""
Creates datasets based on certain parameters:
filepath = the path to the csv where the images, bounding boxes and ids are stored
batch_size = the batch size that the data loader working the data set is created with
validation_split = the part of the data set that is split into a separate validation data set, default 1/3
reduction = if the entire data set is used or whether it is reduced, for a reduced set this value has to between 0 and 1, default 1
raw = whether the data set objects themselves shall be returned along with the data loaders, default False
augment = whether the data set shall augment the images, default True
findDoubles = whether the data set shall find ids that appear more than twice beforehand, default False, 
include_unlabelled = whether the unlabelled data of the set shall be included, default True
Returns either a trainloader and a validationloader or additionally also the training dataset and the validation dataset.
"""
def getDatasets(filepath,batch_size,validation_split=1/3,reduction=1,raw=False,augment=True,findDoubles=False,include_unlabelled=True):
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
                if include_unlabelled:
                    set_raw.append([name,bbox,label])
                else:
                    if not(label == "" or label == "new_whale"):
                        set_raw.append([name,bbox,label])
    random.shuffle(set_raw)
    set_raw_len  = int(len(set_raw)*reduction)
    validation_amount = int(set_raw_len*validation_split)
    validation_set = set_raw[:validation_amount]
    training_set = set_raw[validation_amount:set_raw_len]
    whales = WhaleDataset(imagelist=training_set,img_size=size,augment=augment,findDoubles=findDoubles)
    if len(validation_set) > 0:
        vhales = WhaleDataset(imagelist=validation_set,img_size=size,augment=augment,findDoubles=findDoubles)
        val_loader = torch.utils.data.DataLoader(vhales, batch_size=batch_size,  num_workers=2,shuffle=True)
    else:
        vhales = 0
        val_loader = 0
    train_loader = torch.utils.data.DataLoader(whales, batch_size=batch_size,  num_workers=2,shuffle=True)
    if raw:
        return(train_loader,val_loader,whales,vhales)
    else:
        return(train_loader, val_loader)