import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn.functional as F



class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork,self).__init__()
        self.fc1 = nn.Sequential(
          # First Dense Layer
          nn.Linear(32768, 1024),
          nn.ReLU(),
          #nn.Dropout1d(p=0.5)
          # Second Dense Layer
          nn.Linear(1024, 128),
          nn.ReLU(),
          # Final Dense Layer
          nn.Linear(128,2))

    def forward_once(self, x):
          # Forward pass 
          output = self.fc1(x)
          return output

    def forward(self,input1,input2):
        o1 = self.forward_once(input1)
        o2 = self.forward_once(input2)
        return(o1,o2)


class ContrastiveLoss(torch.nn.Module):

      def __init__(self, margin=2.0):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

      def forward(self, output1, output2, label):
            # Find the pairwise distance or eucledian distance of two output feature vectors
            euclidean_distance = F.pairwise_distance(output1, output2)
            # perform contrastive loss calculation with the distance
            loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

            return loss_contrastive


def oneshot(model,img1,img2):
       # Gives you the feature vector of both inputs
       output1,output2 = model(img1.cuda(),img2.cuda())
       # Compute the distance 
       euclidean_distance = F.pairwise_distance(output1, output2)
       #with certain threshold of distance say its similar or not
       return euclidean_distance
       """if euclidean_distance > 0.5:
               return True
       else:
               return False"""

def train(epochs,learning_rate,set1,set2,ids1,ids2):
    model = SiameseNetwork().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    no_improve = 0
    last_loss = 10000000
    model.train()
    for epoch in range(epochs):
        total_train_loss = 0
        for i in range(len(set1)):
            cl = ContrastiveLoss()
            optimizer.zero_grad()
            i1 = torch.from_numpy(np.array([set1[i,:]])).cuda()
            for j in range(len(set2)):
                i2 = torch.from_numpy(np.array([set2[j,:]])).cuda()
                o1,o2 = model(i1,i2)
                label = torch.zeros(1).cuda()
                if ids1[i,-1] == ids2[j,-1]:
                    label = torch.ones(1).cuda()
                loss = cl(o1,o2,label)
                loss.backward()
                optimizer.step()
                total_train_loss += loss
        print("EPOCH " + str(epoch) + " with loss "  + str(total_train_loss.item()))
        if epoch % 5 == 0:
          if total_train_loss.item() < last_loss:
            last_loss = total_train_loss.item()
            torch.save(model,str("siamese_network"))
          else:
            no_improve += 1
          if no_improve > 5:
            break
    torch.save(model,str("siamese_network"))

def findIndices(tr_ids,te_ids,size=500,cutoff=10):
    tr_idx = []
    te_idx = []
    for i in range(size):
        id_ = te_ids[i,-1]
        te_idx.append(i)
        count = 0
        for j in range(len(tr_ids)):
            if id_ == tr_ids[j,-1]:
                tr_idx.append(j)
                count += 1
            if count == cutoff:
                break
    print(len(tr_idx))
    print(len(te_idx))
    return(tr_idx,te_idx)

def trial():
    tr_enc = np.load("../ae/vae_training_encodings.npy")
    tr_ids = np.load("../ae/vae_training_encoding_ids.npy")
    te_enc = np.load("../ae/vae_test_encodings.npy")
    te_ids = np.load("../ae/vae_test_encoding_ids.npy")
    tr_idx,te_idx = findIndices(tr_ids,te_ids,100,10)
    xs_tr_enc = tr_enc[tr_idx]#np.concatenate([tr_enc[tr_idx],tr_enc[-200:]],axis=0)
    xs_tr_ids = tr_ids[tr_idx]#np.concatenate([tr_ids[tr_idx],tr_ids[-200:]],axis=0)
    xs_te_enc = te_enc[te_idx]
    xs_te_ids = te_ids[te_idx]
    train(350,0.0001,xs_tr_enc,xs_te_enc,xs_tr_ids,xs_te_ids)


def match():
    model = torch.load("siamese_network")
    tr_enc = np.load("../ae/vae_training_encodings.npy")
    tr_ids = np.load("../ae/vae_training_encoding_ids.npy")
    te_enc = np.load("../ae/vae_test_encodings.npy")
    te_ids = np.load("../ae/vae_test_encoding_ids.npy")
    tr_idx,te_idx = findIndices(tr_ids,te_ids,100,10)
    xs_tr_enc = tr_enc[tr_idx]#np.concatenate([tr_enc[tr_idx],tr_enc[-200:]],axis=0)
    xs_tr_ids = tr_ids[tr_idx]#np.concatenate([tr_ids[tr_idx],tr_ids[-200:]],axis=0)
    xs_te_enc = te_enc[te_idx]
    xs_te_ids = te_ids[te_idx]
    c_id = []
    w_id = []
    match_dist = []
    no_match_dist = []
    for i in range(len(te_idx)):
      te = torch.from_numpy(np.array([xs_te_enc[i]]))
      for j in range(len(tr_idx)):
        tr = torch.from_numpy(np.array([xs_tr_enc[j]]))
        res = oneshot(model,te,tr)
        #print(res)
        if res < 0.5 and (xs_te_ids[i,-1] == xs_tr_ids[j,-1]):
          c_id.append([xs_te_ids[i],xs_tr_ids[j]])
        elif res < 0.5 and not(xs_te_ids[i,-1] == xs_tr_ids[j,-1]):
          w_id.append([xs_te_ids[i],xs_tr_ids[j]])
        if (xs_te_ids[i,-1] == xs_tr_ids[j,-1]):
          match_dist.append(res.item())
        else:
          no_match_dist.append(res.item())

    print("Correctly matched") 
    print(len(c_id))
    print("Wrongfully matched") 
    print(len(w_id))
    print(np.mean(match_dist))
    print(np.mean(no_match_dist))
    np.save("wrong.npy",w_id)
    np.save("correct.npy",c_id)



torch.cuda.empty_cache()
match()