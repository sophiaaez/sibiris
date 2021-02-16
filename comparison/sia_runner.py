from siamese import optimal_optimisation
from siamese import train
from siamese import top10Siamese

tr_enc_path = "../ae/vae_training_mean_encodings_simple_v3.npy"
tr_ids_path = "../ae/vae_training_mean_ids_simple_v3.npy"
te_enc_path = "../ae/vae_test_mean_encodings_simple_v3.npy"
te_ids_path = "../ae/vae_test_mean_ids_simple_v3.npy"
#train(1000,0.000001,64,tr_enc_path,tr_ids_path,save_path="siamese_network_vae_mean_v3.pth",size1=256,size2=32,validation_split=1/3)
top10Siamese("siamese_network_vae_mean_v3.pth",tr_enc_path,tr_ids_path,te_enc_path,te_ids_path)
#optimal_optimisation()