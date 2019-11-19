"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import os
import pickle
import urllib
import tarfile
import zipfile
import tensorflow as tf

class Dataset(object):

    def __init__(self, batch_size, shuffle=True, train_bins=[], validation_bins=[], test_bins=[], data_path='bins_translated/', seed=None):
        """
        Construct an iterator object over the data

        Inputs:
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        - train: (optional) Boolean, wether to load the train or the validation set
        - data_path: (optional) String, data path where to retrieve data
        - seed: (optional) Integer, seed needed in order to random shuffle the data
        """
	#read data from file
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.train_bins = train_bins
        self.validation_bins = validation_bins
        self.test_bins = test_bins
        self.X_tr, self.y_tr, self.X_val, self.y_val, self.X_test, self.y_test = self._load_training_data()

    def train_iter(self):
        """
        This method returns an iterator of training data shuffled if specified in the constructor
        :return: iterator of training data
        """
        N, B = self.X_tr.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            if (self.seed is not None):
                np.random.seed(self.seed)
            np.random.shuffle(idxs)
            self.X_tr = self.X_tr[idxs]
            self.y_tr = self.y_tr[idxs]

        return iter((self.X_tr[i:i+B], self.y_tr[i:i+B]) for i in range(0, N, B))

    def _load_training_data(self):
        """
        Load all the training-data or validation-data for the amino-acid data-set.
        Returns the sequence and oncogenic, non-oncogenic class-labels.

        Inputs:
        - train: (optional) List of bins to use as validation set
        - validation: (optional) List of bins to use as validation set
        """
        # bin_dfs_train = [pd.read_csv(self.data_path + 'bio_translated_' + str(i) +
        #                        '.csv', header=None) for i in self.train_bins]
        # bin_dfs_val = [pd.read_csv(self.data_path + 'bio_translated_' + str(i) +
        #                        '.csv', header=None) for i in self.validation_bins] 
        # bin_dfs_val = [pd.read_csv(self.data_path + 'bio_translated_' + str(i) +
        #                        '.csv', header=None) for i in self.test_bins]
        
        # Xs_train = [bin_dfs_train[i][0] for i in range(len(bin_dfs_train))]
        # ys_train = [bin_dfs_train[i][1] for i in range(len(bin_dfs_train))]
        # Xs_train = pd.concat(Xs_train, axis=0)
        # ys_train = pd.concat(ys_train, axis=0)
        # X_train = Xs_train.values
        # y_train = ys_train.values

        # Xs_val = [bin_dfs_val[i][0] for i in range(len(bin_dfs_val))]
        # ys_val = [bin_dfs_val[i][1] for i in range(len(bin_dfs_val))]
        # # Todo testare se funziona con un solo bin di validation
        # Xs_val = pd.concat(Xs_val, axis=0)
        # ys_val = pd.concat(ys_val, axis=0)
        # X_val = Xs_val.values
        # y_val = ys_val.values
        X_train, y_train, X_val, y_val, X_test, y_test = None, None, None, None, None, None
        if (len(self.train_bins) >= 1):
            X_train, y_train = self._load_bins(self.train_bins)
        if (len(self.validation_bins) >= 1):
            X_val, y_val = self._load_bins(self.validation_bins) 
        if (len(self.test_bins) >= 1):
            X_test, y_test = self._load_bins(self.test_bins)
         
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _load_bins(self, bins):
        """
        Load bins from file and concatenate them
        :param bins: bin numbers to read from file
        :return: concatenated data read from file
        """
        bin_dfs = [pd.read_csv(self.data_path + 'bio_translated_' + str(i) +
                               '.csv', header=None) for i in bins]
        
        Xs = [bin_dfs[i][0] for i in range(len(bin_dfs))]
        ys = [bin_dfs[i][1] for i in range(len(bin_dfs))]
        Xs = pd.concat(Xs, axis=0)
        ys = pd.concat(ys, axis=0)
        X = Xs.values
        y = ys.values
        
        return X, y
