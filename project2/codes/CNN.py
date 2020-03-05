#!/usr/bin/env python
# coding: utf-8

# In[1]:


############################## CMSC828C PROJECT 2 #######################################################
# Code : CNN for classification of fashion MNIST 
# Author : Saket Seshadri Gudimetla Hanumath
# UID : 116332293
######################################################################################################
# Importing NECESSARY LIBRARIES for part 2
import mnist_reader
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD
import time


# In[2]:


def load_dataset():    
    # loading the train and test variables
    X_train, y_train = mnist_reader.load_mnist('fashion-mnist-master/data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('fashion-mnist-master/data/fashion', kind='t10k')
    
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    # one hot encoding on the y values
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape) 
    
    return X_train, y_train, X_test, y_test   


# In[3]:


def scale_and_norm(X_train, X_test):
    # convert from int to flot
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    # normalize
    X_train_norm = X_train/255.0
    X_test_norm = X_test/255.0
    
    return X_train_norm, X_test_norm


# In[4]:


# define cnn model
def define_model():
    
    dense_units = 100
    num_filters = 64
    
    model = Sequential()
    model.add(Conv2D(num_filters, (3, 3), activation='relu', 
                     kernel_initializer='he_uniform',
                     input_shape=(28, 28, 1)))
    
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))    
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[5]:


def main():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = scale_and_norm(trainX, testX)
    # evaluate model
    model = define_model()
    model.summary()
    
    # comment the below two lines while running as you may get errors 
    #######
    NAME = "cnn-final-model/"
    tensorboard = TensorBoard(log_dir= 'logs/{}'.format(NAME),  profile_batch = 100000000)
    #######
    
    History = model.fit(trainX, trainY, epochs=10, batch_size=32, 
                            validation_data=(testX, testY),
                       callbacks=[tensorboard])
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    
    # Uncomment to see visualize the loss and accuracy plots 
#     plt.figure(figsize=(10, 10))
#     plt.subplot(211)
#     plt.title('Cross Entropy Loss')
#     plt.plot(History.history['loss'], color='blue', label='train')
#     plt.plot(History.history['val_loss'], color='orange', label='test')
#     # # plot accuracy
#     plt.subplot(212)
#     plt.title('Classification Accuracy')
#     plt.plot(History.history['accuracy'], color='blue', label='train')
#     plt.plot(History.history['val_accuracy'], color='orange', label='test')
#     plt.show()


# In[6]:


main()


# In[ ]:





# In[7]:





# In[ ]:




