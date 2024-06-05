
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import random
import numpy as np
import torch
import os
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler
from itertools import combinations
import random
from sklearn.metrics import accuracy_score, f1_score
from data_loader import dataset
from my_model import TransformerEncoder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default = 1)
parser.add_argument('--t', type=int, default = 1)

args = parser.parse_args()
n_ = args.n
t_ = args.t



if(n_ == 1):

    dataset_name = "ItalyPowerDemand"

elif(n_ == 2):

    dataset_name = "HIGGS"

elif(n_ == 3):

    dataset_name = "SUSY"

elif(n_ == 4):

    dataset_name = "german"

elif(n_ == 5):

    dataset_name = "svmguide3"

elif(n_ == 6):

    dataset_name = "magic04"

else:

    dataset_name = "a8a"

if(t_ == 1):

    data_type = "trapezoidal"

else:

    data_type = "variable_p"




num_layers = 6    # Number of transformer layers
dropout = 0.15     # Dropout probability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_fn = nn.CrossEntropyLoss().to(device)

print()
print()
print()
print("*"*100)
print()
print()
print()
print("dataset_name: ", dataset_name)
print("data_type: ", data_type)
print("num_layers: ", num_layers)
print("dropout: ", dropout)
print("device: ", device)
print()
print()
print()
print("*"*100)
print()
print()
print()

def helper(x):

    arr1 = []
    for i,j in enumerate(x):

        if(j != 0):

            arr1.append(i)

    arr2 = list(combinations(arr1, 2))
    arr3 = list(combinations(arr1, 3))

    random.seed(42)
    random.shuffle(arr1)
    random.shuffle(arr2)
    random.shuffle(arr3)

    return arr1, arr2, arr3




def gen_batch(x,batch = 64):

    arr1,arr2, arr3 = helper(x)
    n2 = len(arr2)
    n3 = len(arr3)

    xi = np.expand_dims(x,axis = 0 )

    arr_ = [xi]

    curr = 1
    flag = True


    for i in arr1:

        temp = np.array(x[:])
        temp[i] = 0.0
        temp = np.expand_dims(temp,axis= 0)
        arr_.append(temp)

        curr += 1

        if(curr == batch):

            break

    rem = batch - curr

    rem1 = min(int(rem*0.65),n2)

    rem2 = min(rem - rem1,n3)

    for i in range(rem1):

        temp = np.array(x[:])

        temp[arr2[i][0]] = 0.0
        temp[arr2[i][1]] = 0.0

        temp = np.expand_dims(temp,axis= 0)

        arr_.append(temp)

    for i in range(rem2):

        temp = np.array(x[:])

        temp[arr3[i][0]] = 0.0
        temp[arr3[i][1]] = 0.0
        temp[arr3[i][2]] = 0.0

        temp = np.expand_dims(temp,axis= 0)

        arr_.append(temp)

    arr_ = np.array(arr_)

    random.seed(42)
    random.shuffle(arr_)

    return arr_


def train_model(model,X,y,optimizer,arr ,arr1, pred):

    optimizer.zero_grad()

    out = model(X)

    pi = torch.argmax(out,axis = 1)

    p = pi.detach().cpu().numpy()

    for i in p:

        pred.append(i)

    correct = sum(y==pi).item()

    error = X.shape[0] - correct

    loss = loss_fn(out, y)

    arr1.append(loss.item())

    loss.backward()

    optimizer.step()

    arr.append(error)


def train(X,Y,model,optimizer,batch = 64 ):

    # print()
    # print("Batch size: ", batch)
    # print()

    model = model.train()
    arr = []
    arr1 = []
    pred = []
    target = []
    N = X.shape[0]
    tl = 0
    for i in tqdm(range(N)):

        xi = gen_batch(X[i],batch)

        n = xi.shape[0]


        yi = np.array([Y[i][0] for j in range(n)])

        for k in yi:

            target.append(k)

        xi = torch.tensor(xi,dtype=torch.float)
        yi = torch.tensor(yi)
        yi = yi.type(torch.int64)

        xi = xi.to(device)
        yi = yi.to(device)

        tl += n

        train_model(model,xi, yi,optimizer,arr,arr1,pred)

    l = sum(arr)

    # res = (l * N) / batch * N


    res1 = l*N/tl

    loss = sum(arr1) / len(arr1)



    return loss,res1, pred, target,arr1

def gen_results(P,SEEDS,name,type_):


    if(name == "magic04"):
        n = 5
    elif(name == "SUSY"):
        n = 4
    else:
        n =  6

    for p in P:

        L1 = []
        A1 = []
        macro_f1 = []
        micro_f1  = []
        f1 = []
        accuracy = []

        for i in SEEDS:

            n_base_feat, n_aux_feat,  X_base, X_aux, X_aux_new, aux_mask,X_, Y, label = dataset(name = name,type = type_, aux_feat_prob = p ,seed = i) #"variable_p"


            if(name == "svmguide3" or name == "HIGGS" or name == "a8a"):

                X = []
                for x in X_:

                    temp = np.append(x,[0 for i in range(0,3)])
                    X.append(temp)

                X_ = np.array(X)

            hidden_dim = X_.shape[1]

            model1 = TransformerEncoder(hidden_dim, n, num_layers, dropout)
            model1 = model1.to(device)
            optimizer1 = optim.Adam(model1.parameters(), lr=0.0001)

            a1,l1,prediction,target,loss = train(X_,Y,model1,optimizer1,64)

            macro_f1.append(f1_score(target , prediction, average='macro'))
            micro_f1.append(f1_score(target , prediction, average='micro'))
            f1.append(f1_score(target , prediction, average= None))
            accuracy.append(accuracy_score(target , prediction))


            L1.append(l1)
            A1.append(a1)


        file_name_ = name + "-" + type_ + "loss-"
        path = "/Results/loss-curves/" + file_name_ + str(p) + ".csv"
        d = {"loss" : loss}
        df = pd.DataFrame(d)
        df.to_csv(path,index = False)
        print()
        print("*"*100)
        print()
        print()
        print("*"*100)
        print()
        print()
        print("*"*100)
        print()
        print()
        print("*"*100)
        print()
        print("Prob: ", p)
        print("Avergae1: " , np.mean(L1))
        print("Std1: " , np.std(L1))
        print("Loss: ", np.mean(A1))
        print("std: ", np.std(A1))
        print("macro f1: ", np.mean( macro_f1))
        print("std macro f1: ", np.std(macro_f1))
        print("micro f1: ", np.mean( micro_f1))
        print("std micro f1: ", np.std(micro_f1))
        print("f1: ", f1 )
        print("acuracy: ",np.mean( accuracy))
        print("Std accuracy: ", np.std(accuracy))
        print()
        d = {
            "Error" : [np.mean(L1)],
            "Std-Error" : [np.std(L1)] ,
            "Loss" : [ np.mean(A1)] ,
            "Std-Loss" : [np.std(A1)],
            "macro f1" : [np.mean( macro_f1)],
            "std micro: " :  [np.std(micro_f1)],
            "micro f1" : [ np.mean( micro_f1)] ,
            "std micro f1: " : [np.std(micro_f1)] ,
            "acuracy"  : [np.mean( accuracy)] ,
            "std accuracy: " : [np.std(accuracy)] ,
        }

        df = pd.DataFrame(d)
        file_name_ = name + "-" + type_ + "-"

        path = "/Results/res/" + file_name_ + str(p) + ".xlsx"

        df.to_excel(path,index = False)


        print("*"*100)
        print()
        print()
        print("*"*100)
        print()
        print()
        print("*"*100)
        print()
        print()
        print("*"*100)
        print()

SEEDS = [random.randint(1, 100) for _ in range(10)]
P = [0.5,0.7]

gen_results(P,SEEDS, dataset_name,data_type)

