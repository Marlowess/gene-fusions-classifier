import tensorflow as tf
from tensorflow import keras
import os
import sys
from pprint import pprint
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import auc as auc_test
from models.attlayer import AttentionWeightedAverage
from models.metrics import f1_m, precision_m, recall_m
from utils.early_stopping_by_loss_val import EarlyStoppingByLossVal

class ModelConvBidirect():
    """
    This architecture is based on 
    https://www.researchgate.net/publication/315860646_A_Deep_Learning_Network_for_Exploiting_Positional_Information_in_Nucleosome_Related_Sequences    
    """ 

    def __init__(self, params):
        """
        It initializes the model before the training
        """  

        self.pretrained_model = params.get('pretrained_model', None)
        if self.pretrained_model is not None:
            # pretrained model load params from pickle
            print("loading model")
            train_dir = "/"
            train_dir = train_dir.join(params['pretrained_model'].split("/")[:-1])                                              
            print(train_dir)
            with open(os.path.join(train_dir, "network_params"), 'rb') as params_pickle:
                self.params = pickle.load(params_pickle)
        else:
            ## new model
            self.params = params      

        self.seed = 42
        self.learning_rate = self.params['lr']
        self.batch_size = self.params['batch_size']                  

        # defines where to save the model's checkpoints 
        self.results_base_dir = self.params['result_base_dir']

        # Weight_decay
        weight_decay = self.params['weight_decay']

        # Model architecture
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Masking(mask_value = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0.], input_shape=(self.params['maxlen'], self.params['vocabulary_len'])))
        self.model.add(tf.keras.layers.Conv1D(self.params['conv_num_filter'], self.params['conv_kernel_size'], activation='relu',
                                              kernel_regularizer=tf.keras.regularizers.l2(weight_decay), 
                                              activity_regularizer=tf.keras.regularizers.l2(weight_decay)))
        self.model.add(tf.keras.layers.MaxPool1D())
        self.model.add(tf.keras.layers.Dropout(self.params['dropout_1_rate']))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.params['lstm_units'], return_sequences=False, 
                                                                         kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                                                         recurrent_regularizer=tf.keras.regularizers.l2(weight_decay))))
        self.model.add(tf.keras.layers.Dropout(self.params['dropout_2_rate']))
        self.model.add(tf.keras.layers.Dense(10, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                            activity_regularizer=tf.keras.regularizers.l2(weight_decay)))
        self.model.add(tf.keras.layers.Dropout(self.params['dropout_3_rate']))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid',
                                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                            activity_regularizer=tf.keras.regularizers.l2(weight_decay)))

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

    def fit_early_stopping_by_loss_val(self, X_tr, y_tr, epochs, early_stopping_loss, callbacks_list, validation_data, shuffle=True):
        print(f"early stopping loss{early_stopping_loss}")
        callbacks_list = self._get_callbacks(train=True)
        callbacks_list.append(EarlyStoppingByLossVal(monitor='val_loss', value=early_stopping_loss))
        history = self.model.fit(x=X_tr, y=y_tr, epochs=self.params['epochs'], shuffle=True,
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
