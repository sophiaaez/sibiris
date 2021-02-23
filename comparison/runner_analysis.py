from analysis import createPlot as createPlot
import numpy as np
from skimage import io, util, transform, color,exposure,filters

#Takes the previously randomly chosen whales, takes their encodings, reduces them in dimesion and plots it.
idspath="../ae/vae_training_mean_ids_simple_v3.npy"
encodingpath="../ae/vae_training_mean_encodings_simple_v3.npy"
idlist = ["w_03e2cf","w_691f2f6","w_955bfe2","w_4441671","w_ac33bfe","w_b3ca4b7","w_b938e96","w_bbfce38","w_ae536d6","w_00904a7","w_5077be0","w_5808373","w_8e88f4f","w_144ecb6","w_222ba28","w_ba4f96c","w_733f661","w_36bbd71","w_b7cf3f5","w_f459442","w_c4b02b0","w_778e474","w_1286f9e","w_3b03149","w_62ec5a8","w_6bab2bd","w_e918d4c"]
idset = set(idlist)
createPlot(encodingpath=encodingpath,idspath=idspath,plotpath="VAEanalysisplot_plus20.png",idlist=idlist,include_img_names=False)

#These are the indices of the first seven random whales, whose images are saved in a subfolder as they are viewed by the algorithm
idxlist = [4722, 9676, 4673, 5846, 4391, 8218, 9940, 12220, 12243, 13604, 13672, 3108, 7895, 18869, 59, 9394, 10238, 14114, 14512, 16588, 3576, 3791, 5723, 8132, 11475, 15537, 15823, 1329, 4283, 5815, 7086, 7804, 11557]
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
    io.imsave("./imgs/_" + name, (color.grey2rgb(image)*255).astype(np.uint8))