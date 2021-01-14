from siamese import optimal_optimisation
from siamese import train
from siamese import top10Siamese

tr_enc_path = "../ae/ae_training_encodings_simple_v2.npy"
tr_ids_path = "../ae/ae_training_ids_simple_v2.npy"
te_enc_path = "../ae/ae_test_encodings_simple_v2.npy"
te_ids_path = "../ae/ae_test_ids_simple_v2.npy"
#train(1000,0.00001,64,tr_enc_path,tr_ids_path,save_path="siamese_network_vae_correct_v2_2.pth",size1=256,size2=32,validation_split=1/3)
top10Siamese("siamese_network_vae_correct_v2_2.pth",tr_enc_path,tr_ids_path,te_enc_path,te_ids_path)