import csv
from dataset import WhaleDataset
import numpy as np
import torch
from skimage import io

def getAndSaveOutputs(filepath,network_path=None,amount=100):
    imagelist = []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            name = str(row[0])
            box = (str(row[1])[1:-1]).split(",")
            bbox = [int(b) for b in box]
            imagelist.append([name,bbox])
    dataset = WhaleDataset(imagelist[:amount],512)
    encoding_ids = None
    encodings = np.array([])
    if network_path:
        model = torch.load(network_path)
        if amount > len(dataset):
            amount = len(dataset)
        for i in range(amount):
            img, img_name, _, _ = dataset.getImageAndAll(i)
            output = model.forward(img.float().cuda())
            imagename = img_name.split("/")[-1]
            image  =output[0,0].cpu().detach()
            io.imsave("./trial_run/output_ae/" + imagename, (color.grey2rgb(image)*255).astype(np.uint8))
            print("./trial_run/output_ae/" + imagename)

def getAndSaveEncodings(filepath,filename,ntype,network_path=None):
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
    encodings = np.array([])
    if network_path:
        model = torch.load(network_path)
        for i in range(len(dataset)):
            img, img_name, bbox, tag = dataset.getImageAndAll(i)
            encoding = model.encode(img.float().cuda())[0]
            eco=encoding.detach().cpu().numpy()
            if i == 0:
                encodings = np.array(eco)
                encoding_ids = [[img_name,str(bbox),tag]]
            else:
                encodings = np.append(encodings,eco,axis=0)
                encoding_ids.append([img_name,str(bbox),tag])
        e_ids = np.array(encoding_ids)
        with open(str(ntype + '_' + filename + '_encodings_simple_v2.npy'), 'wb') as f:
            np.save(f, encodings)
        with open(str(ntype + '_' + filename + '_ids_simple_v2.npy'),'wb') as f:
            np.save(f,e_ids)