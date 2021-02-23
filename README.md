# Sibiris - a failed attempt at individual re-identification for humpback whales (a master thesis)

## Abstract: 
✨Magic ✨

Link to thesis: 

---
## Modules and their algorithms:
- Pre-processing Module:
-- YOLOv3
-- RetinaNet
- Encoding Module:
--Autoencoder 
--Variational Autoencoder
- Comparison Module:
-- Nearest Neighbour
-- Multi-layer Perceptron
-- Siamese Network

## Data:
- both data sets used are not included in the github
- to add the kaggle data set, create a subfolder of ./data titled kaggle
- download the data set from here: https://www.kaggle.com/c/humpback-whale-identification/data (both training and test set) 
- add all images from both sets into the kaggle folder

The final weights of the networks are also not in this github, as its capacity is limited. 
The runner*.py files are the ones that run a certain thing, e.g. nearest neighbour comparison of encodings.
