from ae_simple import optimal_optimisation as ae_oo
from vae_simple import evalSet
from vae_simple import optimal_optimisation as vae_oo
from ae_simple import trainNet as  ae_train
from ae_simple_siamese import trainNet as aes_train
from getandsavestuff import getAndSaveEncodings,getAndSaveOutputs
from vae_simple import trainNet as  vae_train

#vae_train(epochs=1000,learning_rate=0.00001,batch_size=8,data_path="../data/trainingset_final_v2.csv",layers=6,layer_size=32,beta=1,save=True)
#getAndSaveEncodings(filepath="../data/trainingset_final_v2.csv",filename="training",ntype="vae",network_path="VAE_earlystopsave_4_simple_v3_1.pth")
#getAndSaveEncodings(filepath="../data/testset_final_v2.csv",filename="test",ntype="vae",network_path="VAE_earlystopsave_4_simple_v3_1.pth")
#evalSet(filepath="../data/testset_final_v2.csv",beta=1,network_path="VAE_earlystopsave_4_simple_v3_1.pth")
#aes_train(epochs=1000,learning_rate=0.0001,batch_size=8,data_path="../data/trainingset_final_v2.csv",layers=5,layer_size=32,save=False)
#ae_oo()
#vae_oo()
#getAndSaveOutputs(filepath="../data/testset_final_v2.csv",network_path="VAE_earlystopsave_4_simple_v3.pth",amount=10)