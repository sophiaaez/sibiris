/from vae_simple import optimal_optimisation as vae_oo
from vae_simple import evalSet as vae_eval
from vae_simple import trainNet as  vae_train
from getandsavestuff import getAndSaveEncodings,getAndSaveOutputs

#Unomment in for optimising the vae architecture
"""trials = 25
vae_oo(trials)"""

#Uncomment for VAE training, encoding saving, test set evaluation and sample output creation.
"""vae_train(epochs=1000,learning_rate=0.00001,batch_size=8,data_path="../data/trainingset_final_v2.csv",layers=6,layer_size=32,beta=1,save=True)
getAndSaveEncodings(filepath="../data/trainingset_small.csv",filename="training_small",ntype="vae",network_path="VAE_earlystopsave_4_simple_v3_1.pth")
getAndSaveEncodings(filepath="../data/testset_small.csv",filename="test_small",ntype="vae",network_path="VAE_earlystopsave_4_simple_v3_1.pth")
evalSet(filepath="../data/testset_final_v2.csv",beta=1,network_path="VAE_earlystopsave_4_simple_v3_1.pth")
getAndSaveOutputs(filepath="../data/testset_final_v2.csv",network_path="VAE_earlystopsave_4_simple_v3_1.pth",amount=10)"""
