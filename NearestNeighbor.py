# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:41:10 2017

"""
import numpy as np

class NearestNeighbor(object):
    #http://cs231n.github.io/classification/
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X, l='L1', k=1):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test)#, dtype = self.ytr.dtype)
        # loop over all test rows
        for i in range(num_test):
            # find the nearest training example to the i'th test example
            if l == 'L1':
                # using the L1 distance (sum of absolute value differences)
                distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            else:
                distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
            #min_index = np.argmin(distances) # get the index with smallest distance -> k == 1
            idx = np.argpartition(distances,k+1)
            #print('idx ', idx)
            count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #for z in idx:
             #   count [self.ytr[z].astype(np.int64)] = count [self.ytr[z].astype(np.int64)] + 1
            for z in range(0, k):
                count [self.ytr[idx[z]].astype(np.int64)] = count [self.ytr[idx[z]].astype(np.int64)] + 1
            #Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
            #print('count ', count)
            Ypred[i] = np.argmax(count) # predict the label of the nearest example  
        return Ypred