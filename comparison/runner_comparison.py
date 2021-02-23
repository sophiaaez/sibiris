from nn import top10NN
from mlp import top10MLP
from siamese import optimal_optimisation,train,top10Siamese

#Runs evaluation of comparison module based on AE encodings for the modified Nearest Neighbour and Various MLP Architectures
tr_enc_path = "../ae/ae_training_encodings_simple_v3.npy"
tr_ids_path = "../ae/ae_training_ids_simple_v3.npy"
te_enc_path = "../ae/ae_test_encodings_simple_v3.npy"
te_ids_path = "../ae/ae_test_ids_simple_v3.npy"
print("ModNN")
top10NN(tr_enc_path,tr_ids_path,te_enc_path,te_ids_path,reduced=None)
layers = [(50,),(100,),(100,50),(200,),(200,100),(400,),(800,),(1600,),(1600,800),(3200,),(3200,1600),(6400,)]
print("MLP")
for l in layers:
    print(l)
    top10MLP(l,tr_enc_path,tr_ids_path,te_enc_path,te_ids_path,activation="relu",reduced=None)
#Trains the Siamese AE and Evaluates iton the test set
train(1000,0.000001,64,tr_enc_path,tr_ids_path,save_path="siamese_network_ae_correct_v3.pth",size1=256,size2=32,validation_split=1/3)
top10Siamese("siamese_network_ae_correct_v3.pth",tr_enc_path,tr_ids_path,te_enc_path,te_ids_path)

#Runs evaluation of comparison module based on VAE mean encodings for the modified Nearest Neighbour and Various MLP Architectures
tr_enc_path = "../ae/vae_training_mean_encodings_simple_v3.npy"
tr_ids_path = "../ae/vae_training_mean_ids_simple_v3.npy"
te_enc_path = "../ae/vae_test_mean_encodings_simple_v3.npy"
te_ids_path = "../ae/vae_test_mean_ids_simple_v3.npy"
print("ModNN")
top10NN(tr_enc_path,tr_ids_path,te_enc_path,te_ids_path,reduced=None)
layers = [(50,),(100,),(100,50),(200,),(200,100),(400,),(800,),(1600,),(1600,800),(3200,),(3200,1600),(6400,)]
print("MLP")
for l in layers:
    print(l)
    top10MLP(l,tr_enc_path,tr_ids_path,te_enc_path,te_ids_path,activation="relu",reduced=None)
#Trains the Siamese VAE and Evaluates iton the test set
train(1000,0.000001,64,tr_enc_path,tr_ids_path,save_path="siamese_network_vae_mean_v3.pth",size1=256,size2=32,validation_split=1/3)
top10Siamese("siamese_network_vae_mean_v3.pth",tr_enc_path,tr_ids_path,te_enc_path,te_ids_path)
