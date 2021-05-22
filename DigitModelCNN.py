# -*- coding: utf-8 -*-
"""
Created on Wed May 19 04:04:09 2021

@author: Gerges Hanna
"""
import matplotlib.pyplot as plt
import numpy as np

from  tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import load_model
import sys
sys.path.append('G:/python project/Digit_Recognition_CNN/')


class DigitModelCNN:

    def setData(self,X_train,y_train_one_hot,X_test,y_test_one_hot):
        self.X_train=X_train
        self.y_train_one_hot=y_train_one_hot
        self.X_test=X_test
        self.y_test_one_hot=y_test_one_hot
        
    def getData(self):
        return self.X_train,self.y_train_one_hot,self.X_test,self.y_test_one_hot
    
    def saveModel(self,model,file_name):
        model.save(file_name)
        
    def loadModel(self,file_name):
        self.model = load_model('G:/python project/Digit_Recognition_CNN/model.h5')
        return self.model
    
    def setModel(self,model):
        self.model=model
    
    def getModel(self):
        return self.model
        
    def digitPlot(self,xImage,yImage="",predicted=""):
        img=xImage.reshape(28,28)
        plt.imshow(img)
        if yImage == "" and predicted != "":
            plt.title("predicted:"+str(predicted))    
        elif yImage != "" and predicted != "" :
            plt.title("predicted:"+str(predicted)+"\n"+"Correct Label: %d "%yImage)
        elif  yImage != "" and predicted == "":
             plt.title("Correct Label: %d "%yImage)  
        plt.show()
    
    def get_predict(self,image):
        predictions = self.model.predict(image.reshape(1, 28, 28, 1))
        predictions=np.argmax(predictions, axis=1)[0]
        print(predictions)
        self.digitPlot(image,predicted=predictions)
        
    def digitModel(self,epochsNumber=3):
        X_train,y_train_one_hot,X_test,y_test_one_hot=self.getData()
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history=model.fit(X_train, y_train_one_hot, validation_data=(X_test, y_test_one_hot), epochs=epochsNumber)
        self.model=model
        return model,history
     
    def getAcc_Loss(self,model):
        X_train,y_train_one_hot,X_test,y_test_one_hot=self.getData()
        loss,accuracy=model.evaluate(X_test,y_test_one_hot,verbose=1)
        return loss, accuracy
    
    
    def plotModel(self,history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()
       
    def prepareDataForPredection(self,data):
        #Normalize the input to be between (0,1)
        data=data/255.0    
        #Reshap the Input 
        data = data.reshape(10000, 28, 28, 1)
        return data        
        
        
         