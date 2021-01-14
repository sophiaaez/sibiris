from ae_simple import optimal_optimisation as ae_oo
from vae_simple import evalSet
from vae_simple import optimal_optimisation as vae_oo
from ae_simple import trainNet as  ae_train
from ae_simple_siamese import trainNet as aes_train
from getandsavestuff import getAndSaveEncodings,getAndSaveOutputs
from vae_simple import trainNet as  vae_train
#evalSet(filepath="../data/testset_final_v2.csv",beta=15,network_path="VAE_earlystopsave_4_simple_v2_2.pth")
#getAndSaveEncodings(filepath="../data/trainingset_final_v2.csv",filename="training",ntype="ae",network_path="AE_earlystopsave_4_simple_v2_2.pth")
#getAndSaveEncodings(filepath="../data/testset_final_v2.csv",filename="test",ntype="ae",network_path="AE_earlystopsave_4_simple_v2_2.pth")
vae_train(epochs=1000,learning_rate=0.00001,batch_size=1,data_path="../data/trainingset_final_v2_overfit.csv",layers=4,layer_size=32,beta=15,save=True)
#aes_train(epochs=1000,learning_rate=0.0001,batch_size=8,data_path="../data/trainingset_final_v2.csv",layers=5,layer_size=32,save=False)
#ae_oo()
#vae_oo()
getAndSaveOutputs(filepath="../data/trainingset_final_v2_overfit.csv",network_path="VAE_earlystopsave_4_simple_v2_overfit.pth",amount=1)