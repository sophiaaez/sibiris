from PIL import Image
import glob
import random


def is_grey_scale(img_path):
    img = Image.open(img_path).convert('RGB')
    w,h = img.size
    for x in range(15):
        i = random.randint(0,w-1)
        j = random.randint(0,h-1)
        r,g,b = img.getpixel((i,j))
        if r != g != b: return False
    return True

def getStats(folder):
    imagelist = glob.glob(folder + "*.jpg")
    imagelist.extend(glob.glob(folder + "*.JPG"))
    imagelist.extend(glob.glob(folder + "*.jpeg"))
    imagelist.extend(glob.glob(folder + "*.JPEG"))
    print(len(imagelist))
    greys = 0
    colors = 0
    for i in imagelist:
        if is_grey_scale(i):
            greys += 1
        else:
            colors +=1
    print(greys)
    print(colors)

getStats("./data/kaggle/")