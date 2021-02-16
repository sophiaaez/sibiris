from analysis import createPlot as createPlot
import numpy as np
from skimage import io, util, transform, color,exposure,filters

idspath="../ae/vae_training_mean_ids_simple_v3.npy"
encodingpath="../ae/vae_training_mean_encodings_simple_v3.npy"
idlist = ["w_03e2cf","w_691f2f6","w_955bfe2","w_4441671","w_ac33bfe","w_b3ca4b7","w_b938e96","w_bbfce38"]
createPlot(encodingpath=encodingpath,idspath=idspath,plotpath="VAEanalysisplot.png",idlist=idlist)

idxlist = [4722, 9676, 4673, 5846, 4391, 8218, 9940, 12220, 12243, 13604, 13672, 3108, 7895, 18869, 59, 9394, 10238, 14114, 14512, 16588, 3576, 3791, 5723, 8132, 11475, 15537, 15823, 1329, 4283, 5815, 7086, 7804, 11557]
"""
enc = np.load(encodingpath)
ids = np.load(idspath)
print(np.array(ids[idxlist[0],1]))
path = "../data/kaggle/"
for i in idxlist:
    name = ids[i,0]
    print(name)
    crop = ids[i,1]
    box = (str(crop)[1:-1]).split(",")
    bbox = [int(b) for b in box]
    x1,y1,x2,y2 = bbox
    image = io.imread(str(path + name),as_gray=True)
    image = image[y1:y2,x1:x2] #CROPPING
    image = transform.resize(image,(512,512))
    io.imsave("./imgs/_" + name, (color.grey2rgb(image)*255).astype(np.uint8))"""