import csv
from random import shuffle, sample

"""
Merges the cropped data at croppedPath with the data labels at labelledPath
Saves the new csv file at the given futurePath .csv included
"""
def mergeData(croppedPath= "./flukes.csv",labelledPath = "./train.csv",futurePath="./labelled_and_cropped.csv"):
    cropped = []
    with open(croppedPath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            cropped.append(row)
    withlabels = []
    with open(labelledPath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            withlabels.append(row)
    croppedwithlabels = []
    for c in cropped[1:]:
        matched = False
        for l in withlabels[1:]:
            if l[0] in c[0]:
                croppedwithlabels.append([l[0],c[1],l[1]])
                matched = True
                break
        if not matched:
            croppedwithlabels.append(c[0],c[1],"")
    with open(futurePath, "w", newline='') as myfile:
        wr = csv.writer(myfile,delimiter=',')
        wr.writerow(["Imagename", "Bounding Box", "Whale ID"])
        for c in croppedwithlabels:
          wr.writerow(c)

"""
Finds a the index of a list entry in labelled with a matching id_. 
If an index is given, the returned index is a different one.
"""
def findMatchID(id_,labelled,index=-1):
    idx = 0
    for i in range(len(labelled)):
        if labelled[i][2] == id_ and not (i == index):
            idx = i
            break
    return idx

"""
Splits the cropped and labelled data into a traininge/validation and a test set with the 
given test_split, default is 0.25. The two csv files are saved at the given output path
"""
def splitData(croppedandlabelledPath="./labelled_and_cropped.csv",test_split=0.25,outputpath="./"):
    labelled = []
    new_whale = []
    other = []
    with open(croppedandlabelledPath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        counter = 0
        for row in spamreader:
            if counter > 0:
                if len(row) == 3 and (not row[-1] == "new_whale") and len(row[2]) > 0:
                    labelled.append(row)
                elif row[-1] == "new_whale":
                    new_whale.append(row)
                else:
                    other.append(row)
            counter = 1
    testset = []
    trainvalset = []
    shuffle(labelled)
    shuffle(new_whale)
    shuffle(other)
    trainvalset.extend(other)
    trainvalset.extend(new_whale[:-5])
    testset.extend(new_whale[-5:]) #take some new_whale images to check
    total_test_amount = int(len(labelled) * test_split)
    unique = {}
    for l in labelled:
        if len(l)>2 and (not l[2] in unique.keys()):
            unique[l[2]] = 1
        elif len(l)>2 and l[2] in unique.keys():
            unique[l[2]] = unique[l[2]] + 1
    doubles = []  
    total = 0
    double = 0      
    for k,v in unique.items():
        if v > 1:
            total += v
            double += 1
            doubles.append(k)
    rest = []
    for k,v in unique.items():
        if v > 1: #if there's more than one
            #find two of them
            x1 = findMatchID(k,labelled)
            x2 = findMatchID(k,labelled,x1)
            l1 = labelled[x1]
            l2 = labelled[x2]
            #add one to each set
            trainvalset.extend([l1])
            testset.extend([l2])
            #delete them
            labelled.remove(l1)
            labelled.remove(l2)
        if v == 1: #if there's only one
            #find it
            x1 = findMatchID(k,labelled)
            l1 = labelled[x1]
            #add it to the trainvalset
            trainvalset.extend([l1])
            #delete it 
            labelled.remove(l1)
    rest_test_amount = total_test_amount - len(testset) #how many more samples do we need for the testset?
    shuffle(labelled)
    testset.extend(labelled[:rest_test_amount])
    trainvalset.extend(labelled[rest_test_amount:])
    shuffle(testset)
    shuffle(trainvalset)
    with open(str(outputpath + "trainingset_final_v2.csv"), "w", newline='') as myfile:
        wr = csv.writer(myfile,delimiter=',')
        for c in trainvalset:
            wr.writerow(c)
    with open(str(outputpath + "testset_final_v2.csv"), "w", newline='') as myfile:
        wr = csv.writer(myfile,delimiter=',')
        for c in testset:
            wr.writerow(c)

"""
Randomly samples a small data set of the sizes trainsetsize and testsetsize from the existing training and test set files..
Ensures that all individuals in the testset also have images in the trainingset.
Saves the two data sets at trainingset_small.csv and testset_small.csv
"""
def sampleSmallData(trainsetsize=1000,testsetsize=100):
    #first read the labelled part of the trainset
    trainset = []
    filepath = "trainingset_final_v2.csv"
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
                    if not(label == "" or label == "new_whale"):
                        trainset.append([name,bbox,label])
    #sample
    train_sampled = sample(trainset,1000)
    #create id_pool of train_sampled ids
    id_pool = set()
    for t in train_sampled:
        id_pool.add(t[2])
    #read testset and collect all with same ids as in train_sampled
    testset = []
    filepath = "testset_final_v2.csv"
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
                    if not(label == "" or label == "new_whale"):
                        #print(label)
                        if label in id_pool:
                            testset.append([name,bbox,label])
    #sample
    test_sampled = sample(testset,100)
    #save
    with open("trainingset_small.csv", "w", newline='') as myfile:
        wr = csv.writer(myfile,delimiter=',')
        for c in train_sampled:
            wr.writerow(c)
    with open("testset_small.csv", "w", newline='') as myfile:
        wr = csv.writer(myfile,delimiter=',')
        for c in test_sampled:
            wr.writerow(c)