from preprocessing import preprocessImages
from embedding import embedFlukes
from comparison import findClosestFluke
import glob

#get input images somehow
imagelist = glob.glob("./data/images/*.jpg") #dummy input
#preprocess them
fins, flukes = preprocessImages(imagelist) #this actually works
#send through to embedding
flukeembeddings = embedFlukes(flukes) #dummy function
#find closest but loop coz user interaction iterative
for f in flukeembeddings:
    closest = findClosestFluke(f,10) #dummy function