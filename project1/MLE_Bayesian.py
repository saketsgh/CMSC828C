#!/usr/bin/env python
# coding: utf-8

# In[43]:


############################## CMSC828C PROJECT 1 #######################################################
# Code : Bayes Classifier along with PCA and LDA
# author : Saket Seshadri Gudimetla Hanumath
# UID : 116332293
######################################################################################################
# Importing NECESSARY LIBRARIES for part (a)
import numpy as np
import mnist_reader
import math
import time

# Importing NECESSARY LIBRARIES for part (c)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.preprocessing import StandardScaler


# In[44]:


# LOADING THE DATASET
def load_dataset():    
    X_train, y_train = mnist_reader.load_mnist('fashion-mnist-master/data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('fashion-mnist-master/data/fashion', kind='t10k')
    return X_train, y_train, X_test, y_test


# In[45]:


# CALCULATING THE MLE ESTIMATES FOR THE GIVEN LABEL/CLASS
def mlestimates(x_train, y_train, cls_name):
    indices_list = np.where(y_train == cls_name)
    x_class = x_train[indices_list[0]]
    mu = np.mean(x_class, axis = 0)
    cov = np.cov(x_class, rowvar=False)
    return mu, cov


# In[46]:


def discriminant_fun(X_test, thetas):
    # returns the discriminant functions computed for all test samples and for each 
    # of the 10 classes. Thus, it has a shape of 10, 10000
    
    x = X_test.T
    g = []
    
    # using discriminant functions g_i(x)
    for theta in range(0, 10):
            mu, cov = thetas[theta]
            mu = mu.reshape(len(mu), 1)
            x_mu = (np.subtract(x, mu)).T
            x_mu_sig = np.dot(x_mu, np.linalg.pinv(cov))
            g_x_theta = np.multiply(x_mu_sig, x_mu)
            g_x_theta = np.sum(g_x_theta, axis = 1)
            g.append(-0.5*g_x_theta)
            
    g = np.array(g)
    return g


# In[47]:


def get_predicted_labels(g):
    y_pred = []
    
    # Max denotes the index of the the class that has the largest magnitude of discriminant function
    # or that has max posterior probability, since Bayes chooses the one with max magnitude. 
    Max = 0
    
    for j in range(0, 10000):
        for i in range(0, 10):
            if g[i, j] > g[Max, j]:
                Max = i
        y_pred.append(Max)
        
    y_pred = np.array(y_pred)
    return y_pred


# In[48]:


def pred_acc(y_test, y_pred):    
    comparison = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] == y_test[i]:
            comparison += 1
    accuracy = comparison/len(y_pred)
    accuracy = accuracy*100
    return accuracy


# In[52]:


def mle_bayes_classifier(X_train, y_train, X_test, y_test):
    start = time.time()
    
    # CALCULATING MEAN AND COVARIANCE OF EACH CLASS 
    # storing it in the list below
    thetas = []
    for i in range(0, 10):
        mu, cov = mlestimates(X_train, y_train, i)
        thetas.append([mu, cov])
    
    # get discriminant function value for all test samples
    g = discriminant_fun(X_test, thetas)
    
    # get predicted labels
    y_pred = get_predicted_labels(g)
    
    # get accuracy of prediction
    accuracy = pred_acc(y_test, y_pred)
    
    end = time.time()
    time_elapsed = end - start
    
    return accuracy, time_elapsed


# In[60]:


def pca_bayes(X_train, y_train, X_test, y_test, var):
    
    # number of parameters that are responsible for var% of total variance
    pca = PCA(var)

    # print("X_train shape originally : ", X_train.shape)
    X_train_pca = pca.fit_transform(X_train)
    # print("X_train shape after PCA : ", X_train_pca.shape)

    # print("X_test shape originally : ", X_test.shape)
    X_test_pca = pca.transform(X_test)
    # print("X_test after PCA : ", X_test_pca.shape)

    accuracy, pca_time = mle_bayes_classifier(X_train_pca, y_train, X_test_pca, y_test)
    
    return [var, X_train_pca.shape[1], X_test_pca.shape[1], accuracy] , pca_time
    


# In[68]:


def lda_bayes(X_train, y_train, X_test, y_test):
    # number of parameters that are responsible for var% of total variance
    lda = LinearDiscriminantAnalysis()

    # print("X_train shape originally : ", X_train.shape)
    X_train_lda = lda.fit_transform(X_train, y_train)
    # print("X_train shape after LDA : ", X_train_new.shape)

#     print("X_test shape originally : ", X_test.shape)
    X_test_lda = lda.transform(X_test)
#     print("X_test after LDA : ", X_test_lda.shape)

    accuracy, lda_time = mle_bayes_classifier(X_train_lda, y_train, X_test_lda, y_test)
    
    return [X_train_lda.shape[1], X_test_lda.shape[1], accuracy], lda_time


# In[62]:


def main():
    
    #################### PART (A) BAYES CLASSIFIER WITH GAUSSIAN ASSUMPTION #################
    # load the dataset
    X_train, y_train, X_test, y_test = load_dataset()
    bayes_acc, bayes_time = mle_bayes_classifier(X_train, y_train, X_test, y_test)
    
    # getting accuracy of bayes classifier before dimensionality reduction
    print("accuracy of bayes classifier without pca and lda ", bayes_acc)
    print("time elapsed for Bayes without reduction in dimensions --> ", bayes_time)
    #################### PART (C) BAYES CLASSIFIER WITH PCA AND LDA ########################
    # Testing different variance values for accuracy
    var_list = [0.95, 0.90, 0.86, 0.80]
    pca_acc_list = []
    
    # test for each variance parameter how bayes classifier and pca performs on the dataset
    for var in var_list:
        acc_list, pca_time = pca_bayes(X_train, y_train, X_test, y_test, var)
        pca_acc_list.append(acc_list)
    
    print("\nAccuracy of Bayes Classifier after PCA -->")
    print("[Variance, Dimensions of Xtrain after PCA, Dimensions of X_test after PCA, Accuracy]")
    print(pca_acc_list)
    print("time elapsed during PCA computation --> ", pca_time)
    # After observing the output its clear that for when dimensions when reduced to 49 and the 
    # variance parameter was 0.86 acc peaked around 80.97%
    
    # performing LDA and calculating accuracy
    lda_acc_list, lda_time = lda_bayes(X_train, y_train, X_test, y_test)
    print("\nAccuracy of Bayes Classifier after LDA -->")
    print("[Dimensions of Xtrain after LDA, Dimensions of X_test after LDA, Accuracy]")
    print(lda_acc_list)
    print("time elapsed during LDA computation --> ", lda_time)
    # accuracy after LDA was found to be 80.46%
    


# In[63]:


main()


# In[ ]:





# In[ ]:




