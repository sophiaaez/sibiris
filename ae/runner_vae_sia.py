from vae_simple_siamese import trainNet as trainNet
from vae_simple_siamese import top10Siamese as top10
from vae_simple_siamese import optimal_optimisation as oo
from vae_simple_siamese import evalSet
from getandsavestuff import getAndSaveEncodings

#uncomment to run optimisation
#oo()

#uncomment to run training
"""epochs = 1000
data_path="../data/trainingset_final_v2.csv"
learning_rate = 0.0001
batch_size = 4
layer_amount = 6
layer_size = 32
factor = 15
trainNet(epochs,learning_rate,batch_size,data_path,layer_amount,layer_size,factor,save=True,trainFrom=False)"""

#uncomment to create encodings
#getAndSaveEncodings(filepath="../data/trainingset_final_v2.csv",filename="training_siamese",ntype="vae",network_path="VAE_earlystopsave_simple_siamese_v3.pth")
#getAndSaveEncodings(filepath="../data/testset_final_v2.csv",filename="test_siamese",ntype="vae",network_path="VAE_earlystopsave_simple_siamese_v3.pth")
#uncomment to evaluate
#evalSet("../data/testset_final_v2.csv",1,"VAE_earlystopsave_simple_siamese_v3.pth")
#top10("VAE_earlystopsave_simple_siamese_v3.pth", "vae_training_siamese_encodings_simple_v3.npy", "vae_training_siamese_ids_simple_v3.npy", "vae_test_siamese_encodings_simple_v3.npy", "vae_test_siamese_ids_simple_v3.npy")
