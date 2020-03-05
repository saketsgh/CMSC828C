#!/usr/bin/env python
# coding: utf-8

# In[28]:


############################## CMSC828C PROJECT 1 #######################################################
# Code : K Nearest Neigbor along with PCA and LDA
# author : Saket Seshadri Gudimetla Hanumath
# UID : 116332293
######################################################################################################
# # # NECESSARY LIBRARIES
import numpy as np
import mnist_reader
import math
import time
from sklearn.neighbors import KNeighborsClassifier

# Importing NECESSARY LIBRARIES for part (c)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.preprocessing import StandardScaler


# In[20]:


# LOADING THE DATASET
def load_dataset():    
    X_train, y_train = mnist_reader.load_mnist('fashion-mnist-master/data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('fashion-mnist-master/data/fashion', kind='t10k')
    return X_train, y_train, X_test, y_test


# In[21]:


def pca_KNN(X_train, y_train, X_test, y_test):
    
    start = time.time()
    # number of parameters that are responsible for var% of total variance
    pca = PCA(0.95)

#     print("X_train shape originally : ", X_train.shape)
    X_train_pca = pca.fit_transform(X_train)
#     print("X_train shape after PCA : ", X_train_pca.shape)

#     print("X_test shape originally : ", X_test.shape)
    X_test_pca = pca.transform(X_test)
#     print("X_test after PCA : ", X_test_pca.shape)
    
    K = [i for i in range(1, 20, 2)]
    accuracy_list = []

    for i, k in enumerate(K):
        knn = KNeighborsClassifier(n_neighbors=k) 
        knn.fit(X_train_pca, y_train)
        accuracy_list.append([knn.score(X_test_pca, y_test), k])
    
    end = time.time()
    time_elapsed = end - start 
    
    return(accuracy_list), time_elapsed


# In[22]:


def lda_KNN(X_train, y_train, X_test, y_test):
    
    start = time.time()
    
    lda = LinearDiscriminantAnalysis()
    
    # print("X_train shape before LDA : ", X_train.shape)
    X_train_lda = lda.fit_transform(X_train, y_train)
#     print("X_train shape after LDA : ", X_train_lda.shape)

#     print("X_test shape before LDA : ", X_test.shape)
    X_test_lda = lda.transform(X_test)
#     print("X_test after LDA : ", X_test_lda.shape)
    
    K = [i for i in range(1, 20, 2)]
    accuracy_list = []

    for i, k in enumerate(K):
        knn = KNeighborsClassifier(n_neighbors=k) 
        knn.fit(X_train_lda, y_train)
        accuracy_list.append([knn.score(X_test_lda, y_test), k])
    
    end = time.time()
    time_elapsed = end - start
    
    return(accuracy_list), time_elapsed


# In[23]:


def main():
    
    # load the dataset
    X_train, y_train, X_test, y_test = load_dataset()
    
    ######################### Part (b) K-NN without PCA and LDA #################################    
    # Use K Nearest Neigbor Classifier for odd values of k ranging from 1 - 20
    
    # Uncomment the code below to see the output for yourself but this section has been commented 
    # since it takes around 2 hrs for the code to produce an output
#     start = time.time()
#     K = [i for i in range(1, 20, 2)]
#     accuracy_list = []

#     for i, k in enumerate(K):
#         knn = KNeighborsClassifier(n_neighbors=k) 
#         knn.fit(X_train, y_train)
#         accuracy_list.append([knn.score(X_test, y_test), k])
#     knn_time = time.time() - start  

    # Initialising the accuracy list with values obtained during the initial run of KNN for 2 
    # hours. It contains accuracy and its corresponding k value. 
    accuracy_list = [[0.8497, 1], [0.8541, 3], [0.8554, 5], [0.854, 7], [0.8519, 9], [0.8495, 11], 
                     [0.8468, 13], [0.8462, 15], [0.8441, 17], [0.8427, 19]]
    
    ######################### Part (c) K-NN with PCA and LDA #################################
    
    pca_acc_list, pca_time = pca_KNN(X_train, y_train, X_test, y_test)
    lda_acc_list, lda_time = lda_KNN(X_train, y_train, X_test, y_test)
    
    print("KNN for various values of k without PCA and LDA ")
    print(accuracy_list)
    print("time elapsed for computing KNN for 10 values of k --> ~ 280 mins")
    
    print("\nKNN for various values of k with PCA")
    print(pca_acc_list)
    print("time elapsed for computing KNN with PCA for 10 values of k --> ", pca_time)
    
    print("\nKNN for various values of k with LDA")
    print(lda_acc_list)
    print("time elapsed for computing KNN with LDA for 10 values of k --> ", lda_time)


# In[24]:


main()


# In[ ]:





# In[27]:





# In[ ]:




