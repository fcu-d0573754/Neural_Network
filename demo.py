# -*- coding: utf-8 -*-

"""
使用keras創建CNN模型
共2層卷積 ，其中隱藏層激活函數為relu，輸出層則使用softmax，梯度下降選用adam
卷積->池化->卷積->池化->鋪平->全連接
經過5次epoch，準確率達95.28%
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# Load MNIST data
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# Transscale 2D->3D
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /=255

# Encode label

y_train = keras.utils.to_categorical(y_train,num_classes =10)
y_test = keras.utils.to_categorical(y_test,num_classes = 10)

# Build Model
model = Sequential()

# 1st Conv
model.add(Conv2D(32,(3,3),input_shape = (28,28,1)))# add 32 filter
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# 2nd Conv
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(512))
model.add(Dense(10,activation='softmax'))

# Pic generator
gen = ImageDataGenerator()
train_generator = gen.flow(x_train,y_train)

# Set paraset
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
 
model.fit_generator(train_generator,steps_per_epoch=32,epochs=5,validation_data=(x_test,y_test))   

# Evaluate model
score = model.evaluate(x_test,y_test,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', 100*score[1])            

#
import numpy as np
y_pred = model.predict(x_test)
y_pred = y_pred.tolist()
y_out = np.argmax(y_pred,axis =1)

import pandas as pd
data = {'label':y_out} 
data_df = pd.DataFrame(data)
data_df.to_csv('answer.csv')



