import tensorflow as tf
import datetime, os
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import backend as K
from tensorflow import keras
import pandas as pd
import numpy as np
import logging
import matplotlib
import math
import argparse
from ModelFactory import ModelFactory
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
from dataset import Dataset
from utils import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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

def main():
    base_dir = 'bioinfo_project'
    subdir = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('--subdir', default=subdir, help='Checkpoints directory')
    parser.add_argument('--validation', default=False, help='Perform Holdout-Validation for hyperparameter selection', action='store_true')
    parser.add_argument('--train', default=False, help='Train model on whole training data and save it', action='store_true')
    parser.add_argument('--test', default=False, help='Test saved model, in the specified subdir, on test bin', action='store_true')
    parser.add_argument('--batch_size', default=10, help='Number of sample for each training step',type=int)
    parser.add_argument('--num_epochs', default=50, help='Number of epochs before halting the training',type=int)
    parser.add_argument('--lr', default=1e-3, help='Learning rate coefficient',type=float)
    parser.add_argument('--sequence_type', choices=['dna', 'protein'], help='Type of sequence to process in the model: "dna" or "protein"', type=str)
    parser.add_argument('--pretrained_model', help='Path where to find the weights of a pre-trained model', type=str, default=None)
    params = parser.parse_args()
    
    parameters_path  = './parameters.json'
    model_params = load_parameters(parameters_path)
    model_params['pretrained_model'] = params.pretrained_model
    # for key, value in json_params.items():
    #     setattr(params, key, value)
    
    results_dir = os.path.join(base_dir, params.subdir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    _init_logger(results_dir)
    logger.info(params)

    if (params.validation == True):
        results_dir_validation = os.path.join(results_dir, 'results_holdout_validation')
        if not os.path.exists(results_dir_validation):
            os.makedirs(results_dir_validation)
        logger.info('Starting holdout-validation')
        holdout(results_dir_validation, params, model_params)
     
    if (params.train == True):
        # Train model on all training data
        train_bins = [1, 2, 3, 4]
        test_bins = [5]
        logger.info('Training model on bins: {}'.format(train_bins))
        data = Dataset(params.batch_size, shuffle=True, train_bins=train_bins, test_bins=test_bins, seed=0)
        tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True, filters='')
        tokenizer.fit_on_texts(data.X_tr)
        model = Model(len(tokenizer.index_word)+1, results_dir, hidden_size=params.embedding_size,
                      lstm_units=params.lstm_units, lr=params.lr, loss='binary_crossentropy',
                      dropout_rate=params.dropout_rate, recurrent_dropout_rate=params.recurrent_dropout_rate,
                      seed=42)
        test_data = preprocess_validation_data(data.X_test, data.y_test, tokenizer) 
        model, history = train(model, data, tokenizer, test_data, params.epochs, results_dir)
        plot_history(results_dir, history, 'loss_train.png')

        model.model.save(os.path.join(results_dir, 'model.hdf5'))
    if (params.test == True):
        train_bins = [1, 2, 3, 4]
        test_bins = [5]
        data = Dataset(params.batch_size, shuffle=True, train_bins=train_bins, test_bins=test_bins, seed=0)
        logger.info('Testing model on bins: {}'.format(test_bins))
        model = Model(log_dir=results_dir, load=True)
        tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True, filters='')
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
    bS = BioSequence(data.X_tr, data.y_tr, data.batch_size, tokenizer)
    history = model.model.fit_generator(bS, epochs=epochs, shuffle=True, validation_data=validation_data)
    return model, history

def holdout(results_dir, params, model_params, history_filename='history.csv'):
    train_bins = [1, 2, 3]
    val_bins = [4]
    logger.info("Training on bins: {}, validation on {}".format(train_bins, val_bins))
    # todo select bin by correct instance indexes instead of reading again from file
    # and re doing preprocessing
    data = Dataset(params.batch_size, shuffle=True, sequence_type=params.sequence_type,
                   train_bins=train_bins, validation_bins=val_bins, seed=0)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True, filters='', char_level=True)
    tokenizer.fit_on_texts(data.X_tr)
    logger.info("Dictionary: {}".format(tokenizer.index_word))
    logger.info("Dictionary len: {}".format(len(tokenizer.index_word)))
    X_tr, y_tr = preprocess_data(data.X_tr, data.y_tr, tokenizer,
                                model_params['maxlen'], False, len(tokenizer.index_word))
    X_val, y_val = preprocess_data(data.X_val, data.y_val, tokenizer,
                                model_params['maxlen'], False, len(tokenizer.index_word))
    model_params['vocabulary_len'] = len(tokenizer.index_word) + 1
    model = ModelFactory.getEmbeddingBiLstmAttentionProtein(model_params)
    callbacks_list = [
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    ),
                    keras.callbacks.ModelCheckpoint(
                        filepath='my_model.h5',
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=0
                    ),
                    keras.callbacks.CSVLogger(history_filename),
                    keras.callbacks.ReduceLROnPlateau(
                        patience=5,
                        monitor='val_loss',
                        factor=0.75,
                        verbose=1,
                        min_lr=5e-6)
    ]
    model.build()     
    history = model.fit(X_tr, y_tr, params.num_epochs, callbacks_list, (X_val, y_val)) 
    
    # model, history = train(model, data, tokenizer, validation_data, params.epochs, results_dir)
    plot_history(results_dir, history, 'loss_train ' + str(train_bins) + 'val ' + str(val_bins) + '.png')
    # scores returns two element: pos0 loss and pos1 accuracy
    scores = model.model.evaluate(X_val, y_val)
    logger.info("{}: {}".format(model.model.metrics_names[1], scores[1] * 100))

if __name__ == "__main__":
    main()
