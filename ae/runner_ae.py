from ae_simple import optimal_optimisation as ae_oo
from ae_simple import evalSet as ae_eval
from ae_simple import trainNet as  ae_train
from getandsavestuff import getAndSaveEncodings,getAndSaveOutputs

#Unomment in for optimising the ae architecture
"""trials = 25
ae_oo(trials)"""

#Uncomment for AE training, encoding saving, test set evaluation and sample output creation.
"""ae_train(epochs=1000,learning_rate=0.00001,batch_size=8,data_path="../data/trainingset_final_v2.csv",layers=5,layer_size=32,save=True)
getAndSaveEncodings(filepath="../data/trainingset_small.csv",filename="training_small",ntype="ae",network_path="AE_earlystopsave_4_simple_v3.pth")
getAndSaveEncodings(filepath="../data/testset_small.csv",filename="test_small",ntype="ae",network_path="AE_earlystopsave_4_simple_v3.pth")
evalSet(filepath="../data/testset_final_v2.csv",network_path="AE_earlystopsave_4_simple_v3.pth")
getAndSaveOutputs(filepath="../data/testset_final_v2.csv",network_path="AE_earlystopsave_4_simple_v3.pth",amount=10)"""
