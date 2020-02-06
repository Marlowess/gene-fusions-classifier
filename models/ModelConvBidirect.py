import tensorflow as tf
from tensorflow import keras
import os
import sys
from pprint import pprint
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import auc as auc_test
from models.attlayer import AttentionWeightedAverage
from models.metrics import f1_m, precision_m, recall_m

class ModelConvBidirect():
    """
    This architecture is based on 
    https://www.researchgate.net/publication/315860646_A_Deep_Learning_Network_for_Exploiting_Positional_Information_in_Nucleosome_Related_Sequences    
    """ 

    def __init__(self, params):
        """
        It initializes the model before the training
        """        

        self.seed = 42
        self.learning_rate = params['lr']
        self.batch_size = params['batch_size']                  

        # defines where to save the model's checkpoints 
        self.results_base_dir = params['result_base_dir']  

        # Model architecture
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Masking(mask_value = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0.], input_shape=(params['maxlen'], params['vocabulary_len'])))
        self.model.add(tf.keras.layers.Conv1D(50, 3, activation='relu',
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-6), 
                                              activity_regularizer=tf.keras.regularizers.l2(1e-6)))
        self.model.add(tf.keras.layers.MaxPool1D())
        self.model.add(tf.keras.layers.SpatialDropout1D(0.5))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(6, return_sequences=False, 
                                                                         kernel_regularizer=tf.keras.regularizers.l2(1e-6),
                                                                         recurrent_regularizer=tf.keras.regularizers.l2(1e-6))))
        self.model.add(tf.keras.layers.Dropout(0.5))
        # self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(10, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(1e-6),
                                            activity_regularizer=tf.keras.regularizers.l2(1e-6)))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid',
                                            kernel_regularizer=tf.keras.regularizers.l2(1e-6),
                                            activity_regularizer=tf.keras.regularizers.l2(1e-6)))

        # Check if the user wants a pre-trained model. If yes load the weights
        if params['pretrained_model'] is not None:
            self.model.load_weights(params['pretrained_model'])
    

    def build(self, logger=None):
        """
        It compiles the model by defining optimizer, loss and learning rate
        """
        optimizer = tf.keras.optimizers.RMSprop(lr=self.learning_rate, clipnorm=1.0)
        self.model.compile(loss='binary_crossentropy',
                            optimizer=optimizer,
                            metrics=['accuracy', f1_m, precision_m, recall_m])
        if (logger is not None):
            self.model.summary(print_fn=lambda x: logger.info(x))
        else:
            self.model.summary()
            
    def fit(self, X_tr, y_tr, epochs, callbacks_list, validation_data, shuffle=True):
        """
        Fit the model with the provided data and returns the results
        Inputs:
        - X_tr: samples
        - y_tr: labels related to the samples
        - epochs: number of epochs before stopping the training
        - callbacks_list
        - validation_data: data the model is validated on each time a epoch is completed
        - shuffle: if the dataset has to be shuffled before being fed into the network

        Outputs:
        - history: it contains the results of the training
        """
        callbacks_list = self._get_callbacks()
        history = self.model.fit(x=X_tr, y=y_tr, epochs=epochs, shuffle=True, batch_size=self.batch_size,
                    callbacks=callbacks_list, validation_data=validation_data)
        trained_epochs = callbacks_list[0].stopped_epoch - callbacks_list[0].patience +1 if callbacks_list[0].stopped_epoch != 0 else epochs
        return history, trained_epochs
    
    def evaluate(self, features, labels):
        """
        It evalutes the trained model onto the provided data
        Inputs:
        - features: sample of data to validate
        - labels: classes the data belong to
        Outputs:
        - loss
        - accuracy
        - f1_score
        - precision
        - recall
        """
        loss, accuracy, f1_score, precision, recall = self.model.evaluate(features, labels, verbose=0)
        metrics_value = [loss, accuracy, f1_score, precision, recall]

        results_dict = dict(zip(self.model.metrics_names, metrics_value))
        return results_dict

    def print_metric(self, name, value):
        print('{}: {}'.format(name, value))

    def save_weights(self):
        pass    

    def fit_generator(self, generator, steps, validation_data=None, shuffle=True, callbacks_list=None):
        history = self.model.fit_generator(generator, steps, shuffle=True, callbacks=self._get_callbacks(train=True),
                                           validation_data=validation_data)
        return history 

    def fit_generator2(self, generator, steps_per_epoch, epochs, validation_data=None, shuffle=True, callbacks_list=None):
        history = self.model.fit_generator(generator, steps_per_epoch, epochs, shuffle=False, callbacks=self._get_callbacks(train=True),
                                           validation_data=validation_data)
        return history
    
    def plot_model(self,) -> None:
        tf.keras.utils.plot_model(self.model, 'model_graph.png', show_shapes=True)

    def _get_callbacks(self, train=True):
        """
        It defines the callbacks for this specific architecture
        """
        callbacks_list = [            
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.results_base_dir, 'my_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            ),
            keras.callbacks.CSVLogger(os.path.join(self.results_base_dir, 'history.csv')),
            keras.callbacks.ReduceLROnPlateau(
                patience=5,
                monitor='val_loss',
                factor=0.75,
                verbose=1,
                min_lr=5e-6)
        ]
        return callbacks_list
