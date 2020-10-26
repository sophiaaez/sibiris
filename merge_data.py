import csv
from random import shuffle


def mergeData(croppedPath= "./data/train_cropped.csv",labelledPath = "./data/train.csv",futurePath="./data/labelled_and_cropped.csv"):
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
        for l in withlabels[1:]:
            if l[0] in c[0]:
                croppedwithlabels.append([l[0],c[1],l[1]])


    with open(futurePath, "w", newline='') as myfile:
        wr = csv.writer(myfile,delimiter=',')
        wr.writerow(["Imagename", "Bounding Box", "Whale ID"])
        for c in croppedwithlabels:
          wr.writerow(c)


def appendData(croppedandlabelledPath="./data/labelled_and_cropped.csv",testpath="./data/test_cropped.csv"):
    cropped = []
    with open(testpath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            cropped.append(row)

    with open(croppedandlabelledPath, "a", newline='') as myfile:
        wr = csv.writer(myfile,delimiter=',')
        for c in cropped:
            name = c[0]
            clean_name = name.split('/')[-1]
            wr.writerow([clean_name,c[1]])

def splitData(croppedandlabelledPath="./data/labelled_and_cropped.csv",test_split=0.25):
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
    all_whales = (len(labelled) + len(new_whale) + len(other))
    total_test_amount = int(all_whales * test_split)
    print(total_test_amount)
    print(len(labelled))
    unique = {}
    for l in labelled:
        if len(l)>2 and (not l[2] in unique.keys()):
            unique[l[2]] = 1
        elif len(l)>2 and l[2] in unique.keys():
            unique[l[2]] = unique[l[2]] + 1
    doubles = []  
    total = 0
    dobule = 0      
    for k,v in unique.items():
        if v > 1:
            total += v
            dobule += 1
            doubles.append(k)
    rest = []
    for l in labelled:
        if len(l)>2 and l[2] in doubles:#if the tag of the label appears more than once,
            testset.extend([l]) #put it into the testset
            doubles.remove(l[2])
        elif len(l)>2 and unique[l[2]] == 1:
            trainvalset.extend([l])
        elif len(l)>2 and unique[l[2]] > 1 and (not l[2] in doubles):
            rest.extend([l])
    print("TESTSETLEN")
    print(len(testset))
    print("REST" + str(len(rest)))
    shuffle(rest)
    testset.extend(rest[:int(0.5*len(rest))])
    trainvalset.extend(rest[int(0.5*len(rest)):])
    print("TESTSETLEN")
    print(len(testset))
    print(len(trainvalset))




    #test_amount_from_labelled = int(total_test_amount * (len(labelled)/(len(labelled)+len(new_whale)))) #proportional to labels
    #test_amount_from_new_whales = int(total_test_amount * (len(new_whale)/(len(labelled)+len(new_whale)))) #proportional to new_whales
    #print(test_amount_from_labelled)
    #print(test_amount_from_new_whales)
    #testset.extend(labelled[:test_amount_from_labelled])
    #testset.extend(new_whale[:test_amount_from_new_whales])
    #trainvalset.extend(labelled[test_amount_from_labelled:])
    #trainvalset.extend(new_whale[test_amount_from_new_whales:])
    shuffle(testset)
    shuffle(trainvalset)
    print(len(testset))
    print(len(trainvalset))
    with open("./data/trainingset_final.csv", "w", newline='') as myfile:
        wr = csv.writer(myfile,delimiter=',')
        for c in trainvalset:
            wr.writerow(c)

    with open("./data/testset_final.csv", "w", newline='') as myfile:
        wr = csv.writer(myfile,delimiter=',')
        for c in testset:
            wr.writerow(c)




#mergeData()
#appendData()
splitData()
