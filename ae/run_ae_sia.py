from ae_simple_siamese import trainNet as trainNet
from ae_simple_siamese import top10Siamese as top10
from ae_simple_siamese import optimal_optimisation as oo


epochs = 1000
data_path="../data/trainingset_final_v2.csv"
learning_rate = 0.0001
batch_size = 4
layer_amount = 6#trial.suggest_int("layer_amount",4,5,1)
layer_size = 32#trial.suggest_categorical("layer_size",[64,128])#,256])
factor = 6#trial.suggest_int("factor",1,13)"""
trainNet(epochs,learning_rate,batch_size,data_path,layer_amount,layer_size,factor,save=True,trainFrom=False)

#top10("AE_earlystopsave_simple_siamese.pth",data_path,"../data/testset_final_v2.csv")