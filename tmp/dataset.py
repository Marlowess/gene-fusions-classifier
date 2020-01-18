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
import threading

class Dataset(object):

    def __init__(self, batch_size, shuffle=True, sequence_type='dna', train_bins=[],
                 validation_bins=[], test_bins=[], data_path='bins_translated/', seed=None):
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
        self.X_tr, self.y_tr, self.X_val, self.y_val, self.X_test, self.y_test = self._load_training_data(sequence_type)

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

    def _load_training_data(self, sequence_type):
        """
        Load all the training-data or validation-data for the amino-acid data-set.
        Returns the sequence and oncogenic, non-oncogenic class-labels.

        Inputs:
        - train: (optional) List of bins to use as validation set
        - validation: (optional) List of bins to use as validation set
        """
        X_train, y_train, X_val, y_val, X_test, y_test = None, None, None, None, None, None
        if (len(self.train_bins) >= 1):
            X_train, y_train = self._load_bins(self.train_bins, sequence_type)
        if (len(self.validation_bins) >= 1):
            X_val, y_val = self._load_bins(self.validation_bins, sequence_type) 
        if (len(self.test_bins) >= 1):
            X_test, y_test = self._load_bins(self.test_bins, sequence_type)
         
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _load_bins(self, bins, sequence_type):
        """
        Load bins from file and concatenate them
        :param bins: bin numbers to read from file
        :return: concatenated data read from file
        """
        bin_dfs = [pd.read_csv(self.data_path + 'bin_' + str(i) +
                               '_translated.csv') for i in bins]

        if (sequence_type == 'dna'):
          Xs = [bin_dfs[i]['Sequences'] for i in range(len(bin_dfs))]
        elif (sequence_type == 'protein'):
          Xs = [bin_dfs[i]['Translated_sequences'] for i in range(len(bin_dfs))]
        else:
          raise ValueError("sequence_type must be 'protein' or 'dna'")
      
        ys = [bin_dfs[i]['Label'] for i in range(len(bin_dfs))]
        Xs = pd.concat(Xs, axis=0)
        ys = pd.concat(ys, axis=0)
        X = Xs.values
        y = ys.values
        
        return X, y