from nn import top10NN
from mlp import top10MLP

tr_enc_path = "../ae/vae_training_encodings_simple_v3.npy"
tr_ids_path = "../ae/vae_training_ids_simple_v3.npy"
te_enc_path = "../ae/vae_test_encodings_simple_v3.npy"
te_ids_path = "../ae/vae_test_ids_simple_v3.npy"

#print("ModNN")
#top10NN(tr_enc_path,tr_ids_path,te_enc_path,te_ids_path,reduced=None)
layers = [(50,),(100,),(100,50),(200,),(200,100),(400,)]
#layers = [(800,),(1600,),(1600,800),(3200,),(3200,1600),(6400,)]
print("MLP")
for l in layers:
    print(l)
    top10MLP(l,tr_enc_path,tr_ids_path,te_enc_path,te_ids_path,activation="relu",reduced=None)