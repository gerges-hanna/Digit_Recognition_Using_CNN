# -*- coding: utf-8 -*-
"""
Created on Wed May 19 04:04:31 2021

@author: Gerges Hanna
"""
from  tensorflow import keras
from tensorflow.keras.utils import to_categorical
import sys
sys.path.append('G:/python project/Digit_Recognition_CNN/')
from DigitModelCNN import DigitModelCNN
import random


if __name__ == "__main__":
    digitModel=DigitModelCNN()
    model=digitModel.loadModel("G:/python project/Digit_Recognition_CNN/epoch3.h5")
    
    #get Test data or any data you like to get prediction
    (_,_),(X_test,_)=keras.datasets.mnist.load_data()
    
    #we need to prepare it before get result
    data=digitModel.prepareDataForPredection(X_test)
    
    #to get random data and get the prediction and the real plot for this data
    for i in range(5):
        digitModel.get_predict(data[random.randint(0,9999)])
        

