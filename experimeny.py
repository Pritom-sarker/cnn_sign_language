import pandas as pd
import tensorflow as tf
import My_Dl_lib as mdl

#---------------- Data processing

train_data=pd.read_csv('data/sign_mnist_train.csv')
test_data=pd.read_csv('data/sign_mnist_test.csv')

# training set----->>
import numpy as np


def one_hot(list,size):
    x=np.zeros((len(list),size))
    j=0
    for i in list:
            x[j,i]=1
            j+=1
    return x
    # print(x.shape)
    # for i in list:
    #
    #     for j in range (0,size):
    #         if j==i:
    #             x[j]=1
    #             print("hello")
    #         break
    #
    # return x



arr=train_data['label'].values.tolist()
train_Y = one_hot(arr,25)

print(train_Y[2],arr[2])