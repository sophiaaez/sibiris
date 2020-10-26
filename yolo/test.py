import csv
import glob

folder = "../data/train/"

imagelist = glob.glob(str(folder + "*.jpg"))
imagelist.extend(glob.glob(str(folder + "*.JPG")))
imagelist.extend(glob.glob(str(folder + "*.jpeg")))
imagelist.extend(glob.glob(str(folder + "*.JPEG")))

path = "./train_cropped.csv"

file = []
with open(path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        file.append(row)

print(len(imagelist))
print(len(file))
matches = 0
for i in imagelist:
    for f in file:
        if i in f:
            imagelist.remove(i)
            file.remove(f)
            matches += 1
            break;
for i in imagelist:
    for f in file:
        if i in f:
            imagelist.remove(i)
            file.remove(f)
            matches += 1
            break;
for i in imagelist:
    for f in file:
        if i in f:
            imagelist.remove(i)
            file.remove(f)
            matches += 1
            break;
for i in imagelist:
    for f in file:
        if i in f:
            imagelist.remove(i)
            file.remove(f)
            matches += 1
            break;
for i in imagelist:
    for f in file:
        if i in f:
            imagelist.remove(i)
            file.remove(f)
            matches += 1
            break;
for i in imagelist:
    for f in file:
        if i in f:
            imagelist.remove(i)
            file.remove(f)
            matches += 1
            break;
for i in imagelist:
    for f in file:
        if i in f:
            imagelist.remove(i)
            file.remove(f)
            matches += 1
            break;
print(len(imagelist))
print(len(file))
print(matches)
with open("./uncropped.csv", "w", newline='') as myfile:
    wr = csv.writer(myfile,delimiter=',')
    for r in imagelist:
      wr.writerow([r])
