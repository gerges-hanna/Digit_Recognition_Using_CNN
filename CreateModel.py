# -*- coding: utf-8 -*-
"""
Created on Wed May 19 04:02:42 2021

@author: Gerges Hanna
"""

from  tensorflow import keras
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import sys
sys.path.append('G:/python project/Digit_Recognition_CNN/')
from DigitModelCNN import DigitModelCNN

digitModel=DigitModelCNN()

# ============Here we need to handle and prepare our data =====================

#1-Load Data
(X_train,Y_train),(X_test,Y_test)=keras.datasets.mnist.load_data()

#2- Normalize the input to be between (0,1)
X_train=X_train/255.0
X_test=X_test/255.0

#3-See Example after gathering and normalizing Data
print(X_test[0])

#4-Reshap the Input 
X_train = X_train.reshape(60000, 28,28,1)
X_test = X_test.reshape(10000, 28, 28, 1)

#5- Encode our labels to set true value equal 1 and others values equal 0
y_train_one_hot = to_categorical(Y_train)
y_test_one_hot = to_categorical(Y_test)

#6- Path our Data after handling to the class we made it
digitModel.setData(X_train, y_train_one_hot, X_test, y_test_one_hot)

# ======================== Here We prepare our model =====================

#7- fit and make model and set number of epoches
model,history=digitModel.digitModel(epochsNumber=3)

#8- Plot the history of training
digitModel.plotModel(history)

#9- Test The model is work correctly (this method will print the prediction and plot the image)
digitModel.get_predict(X_test[556])

#10- Save our model to be able to get it again without do trainig
digitModel.saveModel(model, "G:/python project/Digit_Recognition_CNN/epoch3.h5")


digitModel.getAcc_Loss(model)