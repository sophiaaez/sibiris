import csv
from dataset import WhaleDataset
import numpy as np
import torch
from skimage import io, color

"""
Reads in the images and their bounding boxes located in a csv file at filepath.
Runs a certain amount (default 100) of these cropped images through the network at network_path
and saves the output/recreated image that the network gives in the subfolder "/outputs/"
"""
def getAndSaveOutputs(filepath,network_path=None,amount=100):
    imagelist = []
    code = network_path[:3]
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            name = str(row[0])
            box = (str(row[1])[1:-1]).split(",")
            bbox = [int(b) for b in box]
            imagelist.append([name,bbox])
    dataset = WhaleDataset(imagelist[:amount],512,augment=False)
    encoding_ids = None
    encodings = np.array([])
    if network_path:
        model = torch.load(network_path)
        if amount > len(dataset):
            amount = len(dataset)
        for i in range(amount):
            img, img_name, _, _ = dataset.getImageAndAll(i)
            output = model.forward(img.float().cuda())
            if len(output) > 1:
                output = output[0]
            imagename = img_name.split("/")[-1]
            image  =output[0,0].cpu().detach()
            io.imsave("./outputs/" + code + imagename, (color.grey2rgb(image)*255).astype(np.uint8))
            io.imsave("./outputs/" + imagename,(color.grey2rgb(img[0,0])*255).astype(np.uint8))

"""
Reads in the images and their bounding boxes located in a csv file at filepath.
A certain amount (default 100) images are augmented as the network will receive them.
These images are saved in the "/outputs/" folder with the code "input" at the beginning of the name.
"""
def getAndSaveInputs(filepath,amount=100):
    imagelist = []
    code = "input"
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            name = str(row[0])
            box = (str(row[1])[1:-1]).split(",")
            bbox = [int(b) for b in box]
            imagelist.append([name,bbox])
    dataset = WhaleDataset(imagelist,512,augment=True)
    encoding_ids = None
    if amount > len(dataset):
        amount = len(dataset)
    for i in range(amount):
        img, img_name, _, _ = dataset.getImageAndAll(i)
        imagename = img_name.split("/")[-1]
        image  = dataset[i][0]
        io.imsave("./outputs/" + code + imagename, (color.grey2rgb(image)*255).astype(np.uint8))

"""
Reads in the images and their bounding boxes located in a csv file at filepath.
Runs a certain amount (default 100) of these cropped images through the encoder part of the 
network at network_path. ntype refers to the type of the network and is either "ae" or "vae".
The filename is either "test" or "training" and determines the name of the output file
in which the encodings are stored. 
The encodings are saved in the same order as their ids, therefore, two .npy files are created.
One with ids (structured [imagename, [bounding box], tag/id]) and one with the encodings.
"""
def getAndSaveEncodings(filepath,filename,ntype,network_path): #ntype = networktype e.g. vae or ae. filename = training or test
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
    dataset = WhaleDataset(imagelist,512,augment=False)
    encodings = np.array([])
    if network_path:
        model = torch.load(network_path)
        for i in range(len(dataset)):
            img, img_name, bbox, tag = dataset.getImageAndAll(i)
            if ntype == "ae":
                encoding = model.encode(img.float().cuda())
            elif ntype == "vae":
                _,encoding,_ = model.encode(img.float().cuda())
            eco=encoding.detach().cpu().numpy()
            if i == 0:
                encodings = np.array(eco)
                encoding_ids = [[img_name,str(bbox),tag]]
            else:
                encodings = np.append(encodings,eco,axis=0)
                encoding_ids.append([img_name,str(bbox),tag])
        e_ids = np.array(encoding_ids)
        with open(str(ntype + '_' + filename + '_mean_encodings_simple_v3.npy'), 'wb') as f:
            np.save(f, encodings)
        with open(str(ntype + '_' + filename + '_mean_ids_simple_v3.npy'),'wb') as f:
            np.save(f,e_ids)