# -*- coding: utf-8 -*-
"""FASHION_MNIST_CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11GPZvjEfHw-ymhYkvOfc68EQH6cHyq7z

# IMPORTS and loading data
"""

import tensorflow as tf
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train,x_test = x_train/255.0,x_test/255.0

"""## 1.sample data"""

px.imshow(x_train[1]) # A shirt!!

"""## 2.padding for extra dim"""

x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)
x_train.shape

# all unique classes
k = len(set(y_train))
print(f"number of unique classes : {k}")

"""# Model building using funtional API"""

from tensorflow.keras.layers import Dense,Flatten,Conv2D,Input,Dropout,MaxPool2D
from tensorflow.keras.models import Model

i = Input(shape = x_train[0].shape)
conv1 = Conv2D(24, (3,3) ,padding = "valid", activation = "relu")(i)
pool1 = MaxPool2D(strides = (2,2))(conv1)
conv2 = Conv2D(64, (3,3) ,padding = "valid", activation = "relu")(pool1)
pool2 = MaxPool2D(strides = (2,2))(conv2)
conv3 = Conv2D(128, (3,3) ,padding = "valid", activation = "relu")(pool2)
pool3 = MaxPool2D(strides = (2,2))(conv3)

flat = Flatten()(pool3)

layer1 = Dense(512,activation = "relu")(flat)
drop1 = Dropout(0.2)(layer1)
layer2 = Dense(256,activation = "relu")(drop1)
drop2 = Dropout(0.2)(layer2)
layer3 = Dense(128,activation = "relu")(drop2)
drop3 = Dropout(0.2)(layer3)
out = Dense(k,activation = "softmax")(drop3)

model = Model(i,out)

"""## 1.compile model"""

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

"""## 2.model image(flow-chart)"""

tf.keras.utils.plot_model(model)

## 3. train! train! train! train! train! train! 

history = model.fit(x_train,y_train,batch_size = 34,validation_data=(x_test,y_test), epochs = 100)

"""# metrics and plots

## 1.accuracy
"""

fig = go.Figure()
fig.add_trace(go.Scatter(y = history.history["accuracy"],x = [*range(len(history.history['accuracy']))], name = "train_acc",mode = "lines+markers"))
fig.add_trace(go.Scatter(y = history.history["val_accuracy"],x = [*range(len(history.history['val_accuracy']))], name = "val_acc",mode = "lines+markers"))
fig.update_yaxes(title = "accuracy")
fig.update_xaxes(title = "epochs")
fig.update_layout(title = "accuracy -- train Vs validation")
fig.show()

"""## 2.loss"""

fig = go.Figure()
fig.add_trace(go.Scatter(y = history.history["loss"],x = [*range(len(history.history['loss']))], name = "train_loss",mode = "lines+markers"))
fig.add_trace(go.Scatter(y = history.history["val_loss"],x = [*range(len(history.history['val_loss']))], name = "val_loss",mode = "lines+markers"))
fig.update_yaxes(title = "loss")
fig.update_xaxes(title = "epochs")
fig.update_layout(title = "loss -- train Vs validation")
fig.show()

y_pred = model.predict(x_test)

y_pred_list = [x.index(max(x)) for x in y_pred.tolist()]

"""## 3.confusion_matrix"""

cm = confusion_matrix(y_test,y_pred_list)
print(cm)

import matplotlib.pyplot as plt
plt.figure(figsize = (16,16))
sns.heatmap(cm, annot = True,fmt = "g")

"""# saving model for future"""

model.save('/content/drive/MyDrive/Colab Notebooks/trained_models/mnist/fashion_mnist_cnn_100epochs.h5')

"""# experiemnts (1-1-21)"""

from tensorflow.keras.layers import Dense,Flatten,Conv2D,Input,Dropout,MaxPool2D
from tensorflow.keras.models import Model,load_model
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

i = Input(shape = x_train[0].shape)
conv1 = Conv2D(24, (3,3) ,padding = "valid", activation = "relu")(i)
pool1 = MaxPool2D(strides = (2,2))(conv1)
conv2 = Conv2D(64, (3,3) ,padding = "valid", activation = "relu")(pool1)
pool2 = MaxPool2D(strides = (2,2))(conv2)
conv3 = Conv2D(128, (3,3) ,padding = "valid", activation = "relu")(pool2)
pool3 = MaxPool2D(strides = (2,2))(conv3)
flat = Flatten()(pool3)
# stop till here for feature maps the next lines are for LOGISTIC regression

logistic = Dense(500,activation = "sigmoid")(flat)
out = Dense(k,activation = "softmax")(logistic)
model_2 = Model(i,out)

model_2.compile(optimizer= "adam", loss = "sparse_categorical_crossentropy", metrics= ["accuracy"])
model_2.fit(x_train,y_train, epochs = 30, batch_size = 28, validation_data=(x_test,y_test))

x_train_flat = model_2.predict(x_train)
x_test_flat = model_2.predict(x_test)

x_train_flat[0].shape

LR = XGBClassifier(objective='multi:softmax', num_class = k, n_estimators=300, random_state=177013)
LR.fit(x_train_flat,y_train)
y_pred_LR = LR.predict(x_test_flat)

random_forest = RandomForestClassifier(n_estimators = 300, random_state = 177013)
random_forest.fit(x_train_flat,y_train)
y_pred_RF = random_forest.predict(x_test_flat)

y_pred_RF

cm_2 = confusion_matrix(y_test,y_pred_RF)
plt.figure(figsize = (8,8))
sns.heatmap(cm_2, annot = True,fmt = "g")

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_RF))

cm_3 = confusion_matrix(y_test,y_pred_LR)
plt.figure(figsize = (8,8))
sns.heatmap(cm_3, annot = True,fmt = "g")

print(classification_report(y_test,y_pred_LR))

