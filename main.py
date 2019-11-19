
import tensorflow as tf
import datetime, os
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
import utils as utl
import logging
import matplotlib
import math
import argparse
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from model import Model
import matplotlib.pyplot as plt
from dataset import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main():
    base_dir = 'bioinfo_project'
    subdir = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--subdir', default=subdir, help='checkpoints directory')
    parser.add_argument('--cv', default=False, help='Perform Cross-Validation for hyperparameter selection', action='store_true')
    parser.add_argument('--train', default=False, help='Train model on whole training data and save it', action='store_true')
    parser.add_argument('--test', default=False, help='Test saved model, in the specified subdir, on test bin', action='store_true')
    params = parser.parse_args()
    
    parameters_path  = './parameters.json'
    json_params = utl.load_parameters(parameters_path)
    for key, value in json_params.items():
        setattr(params, key, value)
    
    results_dir = os.path.join(base_dir, params.subdir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    _init_logger(results_dir)
    logger.info(params)
    
    if (params.cv == True):
        # Train the new model with CV
        results_dir_cv = os.path.join(results_dir, 'results_cv')
        if not os.path.exists(results_dir_cv):
            os.makedirs(results_dir_cv)
        logger.info('Starting cross-validation')
        cross_validate(results_dir_cv, params)
    if (params.train == True):
        # Train model on all training data
        train_bins = [1, 2, 3, 4]
        logger.info('Training model on bins: {}'.format(train_bins))
        data = Dataset(params.batch_size, shuffle=True, train_bins=train_bins, seed=0)
        tokenizer = keras.preprocessing.text.Tokenizer(lower=True)
        tokenizer.fit_on_texts(data.X_tr)
        model = Model(len(tokenizer.index_word)+1, results_dir, hidden_size=128,
                      lr=params.lr, loss='binary_crossentropy', dropout_rate=params.dropout_rate,
                      recurrent_dropout_rate=params.recurrent_dropout_rate, seed=42)
        model = train(model, data, tokenizer, None, params.epochs, results_dir)
        model.model.save(os.path.join(results_dir, 'model.hdf5'))
    if (params.test == True):
        train_bins = [1, 2, 3, 4]
        test_bins = [5]
        data = Dataset(params.batch_size, shuffle=True, train_bins=train_bins, test_bins=test_bins, seed=0)
        logger.info('Testing model on bins: {}'.format(test_bins))
        model = Model(log_dir=results_dir, load=True)
        tokenizer = keras.preprocessing.text.Tokenizer(lower=True)
        tokenizer.fit_on_texts(data.X_tr)
        test_data = preprocess_validation_data(data.X_test, data.y_test, tokenizer)
        scores = model.model.evaluate(test_data[0], test_data[1])
        logger.info("Testing accuracy {}: {}".format(model.model.metrics_names[1], scores[1] * 100))
    
def train(model, data, tokenizer, validation_data, epochs, results_dir):
    """
    Train model
    :param model: class Model that wraps an initialized model to train
    :param data: class Dataset containing loaded bins for training
    :param tokenizer: tokenizer already fitted with all the training set
    :param validation_data: validation data already translated with tokenizer
    :param epochs: number of epochs to train
    :results_dir: directory where to save loss plot
    :return model: class Model wrapping a fitted model
    """
    data_generator = gen(data, tokenizer)
    steps_per_epoch = math.ceil(len(data.X_tr)/data.batch_size)
    history = model.model.fit_generator(data_generator, steps_per_epoch, epochs=epochs, validation_data=validation_data)
    plot_history(results_dir, history)
    return model
    
def cross_validate(results_dir, params):
    """
    Perform k-cross validation on the training bins and validate on the validation one
    :param results_dir: directory where to save model log and loss plot
    :params: command line and json parameters
    """
    num_bins = 5
    test_bin = 5
    bins = [x for x in range(1, num_bins+1) if x != test_bin]
    logger.info('{} bins Cross-Validation'.format(num_bins))
    cv_accuracy = []
    for i in bins:
        train_bins = [x for x in bins if x != i]
        val_bins = [i]
        logger.info("Training on bins: {}, validation on {}".format(train_bins, val_bins))
        # todo select bin by correct instance indexes instead of reading again from file
        # and re doing preprocessing
        data = Dataset(params.batch_size, shuffle=True, train_bins=train_bins, validation_bins=val_bins, seed=0)
        tokenizer = keras.preprocessing.text.Tokenizer(lower=True)
        tokenizer.fit_on_texts(data.X_tr)
        validation_data = preprocess_validation_data(data.X_val, data.y_val, tokenizer)
        model = Model(len(tokenizer.index_word)+1, results_dir, hidden_size=128,
                      lr=params.lr, loss='binary_crossentropy', dropout_rate=params.dropout_rate,
                      recurrent_dropout_rate=params.recurrent_dropout_rate, seed=42)
        train(model, data, tokenizer, validation_data, params.epochs, results_dir)
        
        # scores returns two element: pos0 loss and pos1 accuracy 
        scores = model.model.evaluate(validation_data[0], validation_data[1])
        logger.info("Fold {}".format(i))
        logger.info("{}: {}".format(model.model.metrics_names[1], scores[1] * 100))
        cv_accuracy.append(scores[1] * 100)
    
    logger.info("Mean accuracy: {0:.3f} (+/- {0:.3f})".format(round(np.mean(cv_accuracy), 3),
                                                              round(np.std(cv_accuracy),3)))
        
def gen(data, tokenizer):
    """
    Generator for preprocessing training data at batch level. Here we translate text in sequences, labels in numbers,
    and we apply padding to sequences with lenght less the the longest one in the batch.
    :param data: class Dataset with loaded training bins
    """
    while(True):
        for X, y in data.train_iter():
            X = tokenizer.texts_to_sequences(X)
            X = keras.preprocessing.sequence.pad_sequences(X)
            y = label_text_to_num(y)
            yield X, y

def label_text_to_num(y):
    """
    Function that converts array of chars in corresponding binary label
    :param y: array of labels
    :return: array of binary labels
    """
    f = lambda x: 0 if x == 'N' else 1
    return np.array([f(y[i]) for i in range(len(y))])
    
def plot_history(dir, history):
    """
    Function that plots loss history and if present validation
    :param dir: directory where to save plot
    :history: history containing loss/losses data/s
    """
    loss = history.history['loss']
    # val_loss = history.history['val_loss']
    val_loss = history.history.get('val_loss', None)
    epochs = range(0, len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    if (val_loss is not None):
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
    else:
        plt.title('Training loss')
    plt.legend()
    plt.savefig(os.path.join(dir, 'loss.png'))

def _init_logger(log_dir):
    """
    Initialize main logger
    :param log_dir: directory where to save log file
    """
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
    # file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'main.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    # stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    # add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info('logger initialization done!')
        
def preprocess_validation_data(X_val, y_val, tokenizer):
    """
    preprocess validation data converting text into number sequences and char labels in binary ones
    :param X_val: unpreprocessed validation data
    :param y_val: unpreprocessed label of validation data
    :param tokenizer: tokenizer already fitted with all the training set
    :return: preprocessed validation data
    """
    X_val = tokenizer.texts_to_sequences(X_val)
    X_val = keras.preprocessing.sequence.pad_sequences(X_val)
    y_val = label_text_to_num(y_val)
    return (X_val, y_val)

if __name__ == "__main__":
    main()
    