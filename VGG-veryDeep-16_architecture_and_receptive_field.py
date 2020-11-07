# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 20:41:57 2020

@author: mahesh
"""


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPool2D, ZeroPadding2D, Dropout
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator

modal = Sequential()

modal.add(Conv2D(64, (3, 3), input_shape=(224, 224, 3),
                 activation='relu', padding='VALID'))
modal.add(Conv2D(64, (3, 3), activation='relu', padding='VALID'))
modal.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
modal.add(Conv2D(128, (3, 3), activation='relu', padding='VALID'))
modal.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
modal.add(Conv2D(256, (3, 3), activation='relu', padding='VALID'))
modal.add(Conv2D(256, (3, 3), activation='relu', padding='VALID'))
modal.add(Conv2D(256, (3, 3), activation='relu', padding='VALID'))
modal.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
modal.add(Conv2D(512, (3, 3), activation='relu', padding='VALID'))
modal.add(Conv2D(512, (3, 3), activation='relu', padding='VALID'))
modal.add(Conv2D(512, (3, 3), activation='relu', padding='VALID'))
modal.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
modal.add(Conv2D(512, (3, 3), activation='relu', padding='VALID'))
modal.add(Conv2D(512, (3, 3), activation='relu', padding='VALID'))
modal.add(Conv2D(512, (3, 3), activation='relu', padding='VALID'))
modal.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
modal.add(Flatten())
modal.add(Dense(units=4096, activation='relu'))
modal.add(Dropout(rate=0.5))
modal.add(Dense(units=4096, activation='relu'))
modal.add(Dropout(rate=0.5))
modal.add(Dense(units=2, activation='softmax'))

modal.summary()

modal.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy', 'mse'])

train_data = ImageDataGenerator(rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)
test_data = ImageDataGenerator(rescale=1./255)
train_set = train_data.flow_from_directory('dataset/training_set',
                                           target_size=(224, 224),
                                           batch_size=32,
                                           class_mode='categorical')

test_set = test_data.flow_from_directory('dataset/test_set',
                                         target_size=(224, 224),
                                         batch_size=32,
                                         class_mode='categorical')
modal.fit_generator(train_set,
                    steps_per_epoch=350,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=350)

y_pred = modal.predict(test_set)
