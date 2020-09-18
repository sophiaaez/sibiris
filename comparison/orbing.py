from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import (match_descriptors, corner_harris, corner_peaks, ORB, plot_matches)
import glob
from random import shuffle
import time
import sys

#get imagelist 
def getImagelist(path,names = None,amount=None):
	imagelist = glob.glob(path + str("*.jpg"))
	if names:
		imagelist = []
		for n in names:
			imagelist.append(path + str(n))
	if amount:
		shuffle(imagelist)
		imagelist = imagelist[:amount]
	#print(imagelist)
	return imagelist

def loadImage(path):
	image = imread(path, as_gray=True)
	return image

def saveKeypointsAndDescriptors(path,keypoints,descriptors):
	#write keypoints into file
	#write descriptors into file
	bla = 0

#import each image and get orb keypoints and descriptors
def createORBembeddings(imagelist):
	for i in imagelist: 
		img = loadImage(i)
		extractor = ORB(n_keypoints=1500)
		extractor.detect_and_extract(img)
		saveKeypointsAndDescriptors(i,extractor.keypoints,extractor.descriptors)

def sortFunc(e):
	return e[1]

def findMatch(image_path,imagelist):
	# setup toolbar
	toolbar_width = 50
	sys.stdout.write("[%s]" % (" " * toolbar_width))
	sys.stdout.flush()
	sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
	#setup comparison
	start = time.time()
	image = loadImage(image_path)
	extractor = ORB(n_keypoints=1500)
	extractor.detect_and_extract(image)
	kpts = extractor.keypoints.copy()
	dscs = extractor.descriptors.copy()
	worst_match_score = 0
	match_list = []
	#compare every list element to input
	for i in range(len(imagelist)):
		img = loadImage(imagelist[i])
		extractor = ORB(n_keypoints=1500)
		extractor.detect_and_extract(img)
		kpts1 = extractor.keypoints
		dscs1 = extractor.descriptors
		matches = match_descriptors(dscs,dscs1, cross_check=True)
		#print(len(matches))
		#if match good enough
		if len(matches) > worst_match_score:
			match_list.append([imagelist[i],len(matches)])
			match_list.sort(key=sortFunc,reverse=True)
			if len(match_list) > 11: #top 10 plus matching itself
				match_list = match_list[:11]
				worst_match_score = match_list[-1][-1]
		if i%round(len(imagelist)/toolbar_width)== 0:
			sys.stdout.write("-")
			sys.stdout.flush()
	sys.stdout.write("]\n")
	print("Finished comparison, took {:.2f}s (aka forever)".format(time.time() - start))
	return match_list


#comparison for new image to database of descriptors
images = ['0001f9222.jpg', '2cccac55a.jpg', 'cad8eabe4.jpg', '00029d126.jpg']
imagelist = getImagelist("../data/train/crops/")
matchlist = findMatch("../data/train/crops/0001f9222.jpg",imagelist)
print(matchlist)