# A Novel Approach To Individual Re-Identification Using Semi-Supervised Learning

## Abstract: 
✨Magic ✨

Link to thesis: 

---
## Modules and their algorithms:
- Pre-processing Module: YOLOv3, RetinaNet
- Encoding Module: Autoencoder, (Disentangled-)Variational Autoencoder, Autoencoder connected with Siamese Network at bottleneck, Variational Autoencoder connected with Siamese Network at bottleneck
- Comparison Module: Nearest Neighbour, Multi-layer Perceptron, Siamese Network

## Data:
- both data sets used are not included in the github
- to add the kaggle data set, create a subfolder of ./data titled kaggle
- download the data set from here: https://www.kaggle.com/c/humpback-whale-identification/data (both training and test set) 
- add all images from both sets into the kaggle folder

## Results:
- mAP: 22.5 for YOLOv3, 14.5 for RetinaNet
- MSE Loss: 0.97 for Autoencoder, 1.84 for Variational Autoencoder
- Top 10 Accuracy for Autoencoder Encodings: 0.055 for Nearest Neighbour, 0.024 for best Multi-layer Perceptron, 0.021 for disconnect Siamese network, 0.168 for connected Siamese network
- Top 10 Accuracy for Variational Autoencoder Encodings: 0.05 for Nearest Neighbour, 0.12 for best Multi-layer Perceptron, 0.011 for disconnect Siamese network, 0.17 for connected Siamese network

-> Best approach: YOLOv3 for pre-processing, encoding and comparison by Variational Autoencoder connected with Siamese network

The final weights of the networks are also not in this github, as its capacity is limited. 
The runner*.py files are the ones that run a certain thing, e.g. nearest neighbour comparison of encodings.
