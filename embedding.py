import torch
from skimage import io, util, transform, color

ae_path = "./ae/AE_earlystopsave_4_simple_v3.pth"
img_size = 512

def embedFlukes(flukes):
    embeddings = []
    model = torch.load(ae_path)
    model.eval()
    for f in flukes:
        i = f[0] #flukes is structured [path,[bounding box]]
        x1,y1,x2,y2 = f[1]
        print(i)
        #load image
        image = io.imread(str(i),as_gray=True)
        image = image[y1:y2,x1:x2] #CROPPING
        image = transform.resize(image,(512,512))
        #do embedding
        embed = model.encode(image)
        embeddings.append([i,embed])
    return embeddings

    