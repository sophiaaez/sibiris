from vae_simple import trainNet as vaetrain
from vae_simple import evalSet as vaeeval
from vae_simple import getAndSaveEncodings as vaeencode
from vae_simple import getAndSaveOutputs as vaeoutputs
from ae_simple import trainNet as aetrain
from ae_simple import evalSet as aeeval
from ae_simple import getAndSaveEncodings as aeencode
from ae_simple import getAndSaveOutputs as aeoutputs

"""
layer_sizes = [32,64,128,256]
layer_amounts = [4,5,6]
betas = [1,4,16,32]
for la in layer_amounts:
    for ls in layer_sizes:
        print("Now training AE with " + str(la) + " layers and layersize at bottleneck " + str(ls))
        aetrain(epochs=1000,learning_rate=0.0001,batch_size=8,data_path="../data/trainingset_final.csv",layers=la,layer_size=ls,save=True)
        for b in betas:
            print("Now training VAE with " + str(la) + " layers and layersize at bottleneck " + str(ls) " and beta " + str(b))
            vaetrain(epochs=1000,learning_rate=0.0001,batch_size=8,data_path="../data/trainingset_final.csv",layers=la,layer_size=ls,beta=b,save=True)
for l in range(4,7):
    print("Now training VAE with " + str(l) + " layers.")
    vaetrain(epochs=1000,learning_rate=0.0001,batch_size=8,data_path="../data/trainingset_final.csv",layers=l)
    print("Now training AE with " + str(l) + " layers.")
    aetrain(epochs=1000,learning_rate=0.0001,batch_size=8,data_path="../data/trainingset_final.csv",layers=l)"""

"""
aetrain(epochs=1000,learning_rate=0.0001,batch_size=8,data_path="../data/trainingset_final.csv",layers=4,layer_size=64,save=True)
aeeval("../data/testset_final.csv","AE_earlystopsave_4_simple.pth")
aeencode("../data/testset_final.csv","test","AE_earlystopsave_4_simple.pth")
aeencode("../data/trainingset_final.csv","training","AE_earlystopsave_4_simple.pth")
aeoutputs("../data/trainingset_final.csv","AE_earlystopsave_4_simple.pth",50)

vaetrain(epochs=1000,learning_rate=0.0001,batch_size=8,data_path="../data/trainingset_final.csv",layers=4,layer_size=32,beta=11,save=True)
vaeeval("../data/testset_final.csv","VAE_earlystopsave_4_simple.pth",11)
vaeencode("../data/testset_final.csv","test","VAE_earlystopsave_4_simple.pth")
vaeencode("../data/trainingset_final.csv","training","VAE_earlystopsave_4_simple.pth")
vaeoutputs("../data/trainingset_final.csv","VAE_earlystopsave_4_simple.pth",50)"""




from vae_complex import trainNet as vaetrain
from vae_complex import evalSet as vaeeval
from vae_complex import getAndSaveEncodings as vaeencode
from vae_complex import getAndSaveOutputs as vaeoutputs
from vae_complex import matchTop10 as vaeTop10
from ae_complex import trainNet as aetrain
from ae_complex import evalSet as aeeval
from ae_complex import getAndSaveEncodings as aeencode
from ae_complex import getAndSaveOutputs as aeoutputs
from ae_complex import matchTop10 as aeTop10


"""layer_sizes = [32,64,128,256]
layer_amounts = [4,5,6]
betas = [1,4,16,32]
for la in layer_amounts:
    for ls in layer_sizes:
        print("Now training AE with " + str(la) + " layers and layersize at bottleneck " + str(ls))
        aetrain(epochs=1000,learning_rate=0.0001,batch_size=8,data_path="../data/trainingset_final.csv",layers=la,layer_size=ls,save=True)
        for b in betas:
            print("Now training VAE with " + str(la) + " layers and layersize at bottleneck " + str(ls) " and beta " + str(b))
            vaetrain(epochs=1000,learning_rate=0.0001,batch_size=8,data_path="../data/trainingset_final.csv",layers=la,layer_size=ls,beta=b,save=True)"""
"""for l in range(4,7):
    print("Now training VAE with " + str(l) + " layers.")
    vaetrain(epochs=1000,learning_rate=0.0001,batch_size=8,data_path="../data/trainingset_final.csv",layers=l)
    print("Now training AE with " + str(l) + " layers.")
    aetrain(epochs=1000,learning_rate=0.0001,batch_size=8,data_path="../data/trainingset_final.csv",layers=l)"""
aetrain(epochs=1000,learning_rate=0.0001,batch_size=8,data_path="../data/trainingset_final.csv",layers=4,layer_size=64,save=True)
aeeval("../data/testset_final.csv","AE_earlystopsave_4_complex.pth")
aeencode("../data/testset_final.csv","test","AE_earlystopsave_4_complex.pth")
aeencode("../data/trainingset_final.csv","training","AE_earlystopsave_4_complex.pth")
aeoutputs("../data/trainingset_final.csv","AE_earlystopsave_4_complex.pth",20)

#vaetrain(epochs=1000,learning_rate=0.0001,batch_size=8,data_path="../data/trainingset_final.csv",layers=4,layer_size=32,beta=11,save=True)
#vaeeval("../data/testset_final.csv","VAE_earlystopsave_4",11)
#vaeencode("../data/testset_final.csv","VAE_earlystopsave_4.pth")
#vaeencode("../data/trainingset_final.csv","VAE_earlystopsave_4.pth")
#vaeoutputs("../data/trainingset_final.csv","VAE_earlystopsave_4.pth",20)
#aeTop10()
#vaeTop10()

"""
from vae_complex_dec import trainNet as vaetrain
from vae_complex_dec import evalSet as vaeeval
from vae_complex_dec import getAndSaveEncodings as vaeencode
from vae_complex_dec import getAndSaveOutputs as vaeoutputs
from vae_complex_dec import matchTop10 as vaeTop10
from ae_complex_dec import trainNet as aetrain
from ae_complex_dec import evalSet as aeeval
from ae_complex_dec import getAndSaveEncodings as aeencode
from ae_complex_dec import getAndSaveOutputs as aeoutputs
from ae_complex_dec import matchTop10 as aeTop10"""



#vaetrain(epochs=1000,learning_rate=0.0001,batch_size=8,data_path="../data/trainingset_final.csv",layers=4,layer_size=32,beta=11,save=True)
#vaeeval("../data/testset_final.csv","VAE_earlystopsave_4",11)
#vaeencode("../data/testset_final.csv","VAE_earlystopsave_4.pth")
#vaeencode("../data/trainingset_final.csv", "training","VAE_earlystopsave_4_dec.pth")
#vaeencode("../data/testset_final.csv","test","VAE_earlystopsave_4_dec.pth")
#vaeoutputs("../data/trainingset_final.csv","VAE_earlystopsave_4_dec.pth",20)
#aeTop10()

#vaeTop10()
#aetrain(epochs=1000,learning_rate=0.0001,batch_size=4,data_path="../data/trainingset_final.csv",layers=4,layer_size=64,save=True)
#aeencode("../data/trainingset_final.csv", "training","AE_earlystopsave_4_dec.pth")
#aeencode("../data/testset_final.csv","test","AE_earlystopsave_4_dec.pth")
#aeeval("../data/testset_final.csv","AE_earlystopsave_4")
#aeencode("../data/testset_final.csv","AE_earlystopsave_4.pth")
#aeencode("../data/trainingset_final.csv","AE_earlystopsave_4.pth")
#aeoutputs("../data/trainingset_final.csv","AE_earlystopsave_4.pth",20)
#aeTop10()

#from ae_simple_stn import trainNet as aetrain
#from ae_simple_stn import getAndSaveEncodings as aetrnencode
#aetrain(epochs=1000,learning_rate=0.0001,batch_size=4,data_path="../data/trainingset_final.csv",layers=4,layer_size=64,save=True)
#aetrnencode("../data/trainingset_final.csv", "training","AE_earlystopsave_4_dec.pth")
#aetrnencode("../data/testset_final.csv","test","AE_earlystopsave_4_dec.pth")