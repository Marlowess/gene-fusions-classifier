import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Masking, Bidirectional, LSTM, Dropout, Flatten
from tensorflow.keras.regularizers import l2, l1_l2
import pickle
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
from utils.early_stopping_by_loss_val import EarlyStoppingByLossVal
import json

class ModelBidirectDNA():
    def __init__(self, params):
        """
        It initializes the model before the training
        """        

        # defines where to save the model's checkpoints 
        self.results_base_dir = params['result_base_dir'] 

        self.pretrained_model = params.get('pretrained_model', None)    
        if self.pretrained_model is not None:
            # pretrained model load params from pickle
            print("loading model")
            train_dir = "/"
            train_dir = train_dir.join(params['pretrained_model'].split("/")[:-1])                                              
            print(train_dir)
            with open(os.path.join(train_dir, "network_params"), 'rb') as params_pickle:
                self.params = pickle.load(params_pickle)
            self.params['result_base_dir'] = self.results_base_dir
        else:
            ## new model
            self.params = params 

        self.seeds = [42, 101, 142, 23, 53]
        self.learning_rate = self.params['lr']
        self.batch_size = self.params['batch_size']
        weight_decay = self.params['weight_decay']                  
 
        # Architecture --- emoji network
        weight_init = tf.keras.initializers.glorot_uniform
        recurrent_init = tf.keras.initializers.orthogonal(seed=42)

        # Model definition
        self.model = Sequential()
        self.model.add(Masking(mask_value = [1., 0., 0., 0., 0.], 
            input_shape=(self.params['maxlen'], self.params['vocabulary_len'])))
        self.model.add(tf.keras.layers.Conv1D(self.params['conv_num_filter'], self.params['conv_kernel_size'], activation='relu',
                                              kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                              kernel_initializer=weight_init(self.seeds[2]), 
                                              activity_regularizer=tf.keras.regularizers.l2(weight_decay)))
        self.model.add(tf.keras.layers.MaxPool1D())
        self.model.add(tf.keras.layers.Dropout(self.params['dropout_1_rate'], seed=self.seeds[0]))
        self.model.add(tf.keras.layers.Conv1D(self.params['conv_num_filter'], self.params['conv_kernel_size'], activation='relu',
                                              kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                              kernel_initializer=weight_init(self.seeds[3]), 
                                              activity_regularizer=tf.keras.regularizers.l2(weight_decay)))
        self.model.add(tf.keras.layers.MaxPool1D())        
        self.model.add(Bidirectional(LSTM((int)(self.params['lstm_units']), return_sequences=False,
                                                            dropout=self.params['lstm_input_dropout'],
                                                            kernel_initializer=weight_init(self.seeds[0]),
                                                            recurrent_initializer=recurrent_init,
                                                            kernel_regularizer=l2(self.params['weight_decay'])
                                                                )))
        self.model.add(Dropout(self.params['lstm_output_dropout'], seed=self.seeds[2]))
        self.model.add(Dense(8, activation='relu', kernel_initializer=weight_init(self.seeds[0])))
        self.model.add(Dropout(self.params['dense_dropout_rate'], seed=self.seeds[3]))
        self.model.add(Dense(1, activation='sigmoid',
                                            kernel_initializer=weight_init(self.seeds[4]),
                                            kernel_regularizer=l2(self.params['weight_decay'])))

        # Check if the user wants a pre-trained model. If yes load the weights
        if self.pretrained_model is not None:
            self.model.load_weights(self.pretrained_model)
    

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

        # Print params onto the logger
        if logger is not None:
            logger.info("\n" + json.dumps(self.params, indent=4))
            
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
        # print(X_tr.shape)
        # X_tr = np.reshape(X_tr, (X_tr.shape[0], X_tr.shape[1], -1))

        # X_val = validation_data[0]
        # y_val = validation_data[1]

        # X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], -1))

        # validation_data = (X_val, y_val)
        
        history = self.model.fit(x=X_tr, y=y_tr, epochs=epochs, shuffle=True, batch_size=self.batch_size,
                    callbacks=callbacks_list, validation_data=validation_data)
        trained_epochs = callbacks_list[0].stopped_epoch - callbacks_list[0].patience +1 if callbacks_list[0].stopped_epoch != 0 else epochs
        return history, trained_epochs
    
    def fit_early_stopping_by_loss_val(self, X_tr, y_tr, epochs, early_stopping_loss, callbacks_list, validation_data, shuffle=True):
        print(f"early stopping loss: {early_stopping_loss}")
        callbacks_list = self._get_callbacks(train=True)
        callbacks_list.append(EarlyStoppingByLossVal(monitor='val_loss', value=early_stopping_loss))
        history = self.model.fit(x=X_tr, y=y_tr, epochs=epochs, batch_size=self.batch_size, shuffle=True,
                    callbacks=callbacks_list, validation_data=validation_data)        
        return history
    
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
        with open(os.path.join(self.results_base_dir, "network_params"), 'wb') as params_pickle:
            pickle.dump(self.params, params_pickle)

        self.model.save_weights(os.path.join(self.results_base_dir, 'my_model_weights.h5'))
        model_json = self.model.to_json()
        with open(os.path.join(self.results_base_dir, "model.json"), "w") as json_file:
            json_file.write(model_json)    

    def fit_generator(self, generator, steps, validation_data=None, shuffle=True, callbacks_list=None):
        history = self.model.fit_generator(generator, steps, shuffle=True, callbacks=self._get_callbacks(train=True),
                                           validation_data=validation_data)
        return history 

    def fit_generator2(self, generator, steps_per_epoch, epochs, validation_data=None, shuffle=True, callbacks_list=None):
        history = self.model.fit_generator(generator, steps_per_epoch, epochs, shuffle=False, callbacks=self._get_callbacks(train=True),
                                           validation_data=validation_data)
        return history
        

    def _get_callbacks(self, train=True):
        """
        It defines the callbacks for this specific architecture
        """
        callbacks_list = [            
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.results_base_dir, 'model_checkpoint_weights.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            ),
            keras.callbacks.CSVLogger(os.path.join(self.results_base_dir, 'history.csv')),
            keras.callbacks.ReduceLROnPlateau(
                patience=10,
                monitor='val_loss',
                factor=0.75,
                verbose=1,
                min_lr=5e-6)
        ]
        return callbacks_list

    def predict(self,  x_test, batch_size: int = 32, verbose: int = 0) -> np.array:
        # return np.asarray([])
        return self.model.predict(
            x_test,
            batch_size=batch_size,
            verbose=verbose,
            ).ravel()

    def predict_classes(self,  x_test, batch_size: int = 32, verbose: int = 1) -> np.array:
        # return np.asarray([])
        try:
            return self.model.predict_classes(x_test)
        except Exception as err:
            print(f"EXCEPTION-RAISED: {err}")
            sys.exit(-1)
        pass