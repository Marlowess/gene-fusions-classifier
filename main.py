
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
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from model import Model
import matplotlib.pyplot as plt
from dataset import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main():
    base_dir = 'bioinfo_project'
    sub_dir = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S')
    results_dir = base_dir + '/results_cv/' + sub_dir
    # training_set_path = '../data/bio_translated_training.csv'
    # parameters_path  = './parameters.json'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    _init_logger(results_dir)
    
    batch_size = 32
    # logger.info('preprocessing data')
    # data = Dataset(batch_size, shuffle=True, train_bins=[1, 2, 3], validation_bins=[4], seed=0)
    # tokenizer = keras.preprocessing.text.Tokenizer(lower=True)
    # tokenizer.fit_on_texts(data.X_tr)
    # val = preprocess_validation_data(data.X_val, data.y_val, tokenizer)
    
    # Create a new model architecture
    # model_name = "Unidirectional_LSTM-proteins_fusion with Dropout"

    # model = Model(len(tokenizer.index_word)+1, results_dir, model_name, hidden_size=128,
                #   lr=0.001, loss='binary_crossentropy',seed=42)
    # train(model, data, tokenizer, val, results_dir)
    # Save the model
    # tf.keras.models.save_model(model, model_dir + '/model_' + subdir + '.hdf5')

    # Train the new model with CV
    cross_validate(32, results_dir)
    
def train(model, data, tokenizer, validation_data, results_dir):
    data_generator = gen(data, tokenizer)
    steps_per_epoch = math.ceil(len(data.X_tr)/data.batch_size)
    history = model.model.fit_generator(data_generator, steps_per_epoch, epochs=1, validation_data=validation_data)
    plot_history(results_dir, history)
    
def cross_validate(batch_size, results_dir):
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
        data = Dataset(batch_size, shuffle=True, train_bins=train_bins, validation_bins=val_bins, seed=0)
        tokenizer = keras.preprocessing.text.Tokenizer(lower=True)
        tokenizer.fit_on_texts(data.X_tr)
        validation_data = preprocess_validation_data(data.X_val, data.y_val, tokenizer)
        model = Model(len(tokenizer.index_word)+1, results_dir, hidden_size=128,
                      lr=0.001, loss='binary_crossentropy',seed=42)
        train(model, data, tokenizer, validation_data, results_dir)
        
        # scores returns two element: pos0 loss and pos1 accuracy 
        scores = model.model.evaluate(validation_data[0], validation_data[1])
        logger.info("Fold {}".format(i))
        logger.info("{}: {}".format(model.model.metrics_names[1], scores[1] * 100))
        cv_accuracy.append(scores[1] * 100)
    
    logger.info("Mean accuracy: {0:.3f} (+/- {0:.3f})".format(round(np.mean(cv_accuracy), 3),
                                                              round(np.std(cv_accuracy),3)))
        
def gen(data, tokenizer):
    while(True):
        for X, y in data.train_iter():
            X = tokenizer.texts_to_sequences(X)
            X = keras.preprocessing.sequence.pad_sequences(X)
            y = label_text_to_num(y)
            yield X, y

def label_text_to_num(y):
    f = lambda x: 0 if x == 'N' else 1
    return np.array([f(y[i]) for i in range(len(y))])
    
def plot_history(dir, history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(0, len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(os.path.join(dir, 'loss.png'))

def _init_logger(log_dir):
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
    X_val = tokenizer.texts_to_sequences(X_val)
    X_val = keras.preprocessing.sequence.pad_sequences(X_val)
    y_val = label_text_to_num(y_val)
    return (X_val, y_val)

if __name__ == "__main__":
    main()
    