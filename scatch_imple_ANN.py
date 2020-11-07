# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:06:16 2020

@author: mahesh
"""
import numpy as np
from tqdm import tqdm
import pandas as pd

#inputs = [np.random.randint(1,10) for i in range(9)]
#y_true = [np.random.randint(1,10) for i in range(9)]


data_set = pd.read_csv('Churn_modelling.csv')
inputs = data_set.iloc[:1000, [3, 6, 7, 8, 12]].values.tolist()
y_true = data_set.iloc[:1000, -1].values.tolist()


# ---------------------------first layer---------------------------------
weights_L1 = [0.01*np.random.randint(9) for i in range(len(inputs[0]))]
weights_dict_L1 = [dict() for x in range(len(weights_L1))]

for x in range(len(weights_L1)):
    weights_dict_L1[x]['W'] = weights_L1[x]
    weights_dict_L1[x]['b1'] = 1

# --------------------------second_layer---------------------------------
weights_L2 = [0.01*np.random.randint(1, 10)
              for i in range(int(len(weights_L1)/2))]
weights_dict_L2 = [dict() for x in range(len(weights_L2))]

for x in range(len(weights_L2)):
    weights_dict_L2[x]['W'] = weights_L2[x]
    weights_dict_L2[x]['b1'] = 1

'''    
#-------------------------third_layer------------------------------------   
weights_L3 = [0.01*np.random.randint(1,10) for i in range(5)] 
weights_dict_L3 = [dict() for x in range(len(weights_L3))]

for x in range(len(weights_L3)):
    weights_dict_L3[x]['W'] = weights_L3[x]
    weights_dict_L3[x]['b1'] = 1
'''

# --------------------------output layer---------------------------------
weights_dict_out = {'W': 0.01*np.random.randint(1, 10), 'b1': 1}


# ------------activation functions---------------------------------------
def relu(x): return max(0, x)
def sigmoid(x): return np.power(1+np.exp(-x), -1)
def Lrelu(x): return max(0.01*x, x)


def maxout(x, y): return max(x, y)

# -----------------------train_layers APIs-------------------------------


def train_start_layer(weights_dict, inputs):
    output = []
    for x in range(len(inputs)):
        output.append(relu(inputs[x]*weights_dict[x]
                           ['W'] + weights_dict[x]['b1']))
    return output


def train_mid_layers(weights_dict, inputs):
    output = []
    for x in range(len(weights_dict)):
        total = 0.0
        for y in range(len(inputs)):
            temp_sum = 0.0
            temp_sum = inputs[y]*weights_dict[x]['W']+weights_dict[x]['b1']
            total += temp_sum
        output.append(Lrelu(total))
    return output


def final_layer(weights_dict, inputs):
    total = 0.0
    for y in range(len(inputs)):
        temp_sum = 0.0
        temp_sum = inputs[y]*weights_dict['W']+weights_dict['b1']
        total += temp_sum
    output = sigmoid(total)
    return output


'''
#--------------------------train modal-checksums---------------------    
first_layer = train_start_layer(weights_dict_L1,inputs)
second_layer = train_mid_layers(weights_dict_L2, first_layer)
y_pred = final_layer(weights_dict_out,second_layer)
'''

# ----------------------------loss functions---------------------------


def hinge_loss(y_pred, y_true):
    loss_mat = []
    loss = 0
    total = 0
    for x in range(len(y_true)):
        loss_mat.append(max(0, y_pred+1-y_true[x]))
        total += loss_mat[x]
    loss = total/len(y_true)
    return loss


def mean_sqaure_loss(y_pred, y_true):
    loss_mat = []
    loss = 0
    total = 0
    for x in range(len(y_true)):
        loss_mat.append()
        total += loss_mat[x]
    loss = total/len(y_true)
    return loss

# ---------------------------back-prop_GDs-----------------------------


def gradient_descent_mse(weights_dict, inputs):
    for i in range(len(weights_dict)):
        deri_W = 0.0
        deri_W = (-2/len(inputs)) * \
            (y_true[i]-(weights_dict[i]['W']*inputs[i] + weights_dict[i]['b1']))
    return deri_W


def gradient_descent_hinge(weights_dict, inputs, y_pred):
    grad = 0
    loss = 0
    for i in range(len(weights_dict)):
        loss += max(0, 1-y_pred)
        grad += 0 if y_pred > 1 else -y_true*inputs[i]
    return loss, grad


def train_modal(inputs, y_true, weights_dict_L1, weights_dict_L2, batchsize=10, epochs=10, lr=0.01):
    loss = []
    for x in tqdm(range(epochs)):
        i = 0
        while(i <= len(inputs)):
            temp_in = inputs[i]
            y_temp = y_true[i]
            first_layer = train_start_layer(weights_dict_L1, temp_in)
            second_layer = train_mid_layers(weights_dict_L2, first_layer)
            #third_layer = train_mid_layers(weights_dict_L3,second_layer)
            y_pred = final_layer(weights_dict_out, second_layer)
            loss.append(hinge_loss(y_pred, y_true))
            weights_dict_L1 = gradient_descent_hinge(
                weights_dict=weights_dict_L1, inputs=temp_in, y_true=y_temp)
            weights_dict_L2 = gradient_descent_hinge(
                weights_dict_L2, inputs=temp_in, y_true=y_temp)
            # weights_dict_L3 = gradient_descent_hinge(weights_dict_L3, inputs = temp_in,y_true = y_temp)
            i = i + 1
        print(loss[x])
    return weights_dict_L1, weights_dict_L2  # ,weights_dict_L3


tl1, tl2, tl3 = train_modal(
    inputs, y_true, weights_dict_L1, weights_dict_L2, epochs=300, lr=0.1)

# -----------prdict-func------------
