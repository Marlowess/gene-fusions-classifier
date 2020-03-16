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
import json

class ModelConvBidirect():
    """
    This architecture is based on 
    https://www.researchgate.net/publication/315860646_A_Deep_Learning_Network_for_Exploiting_Positional_Information_in_Nucleosome_Related_Sequences    
    """ 

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

        self.seed = 42
        self.learning_rate = self.params['lr']
        self.batch_size = self.params['batch_size']                  
 
        # Weight_decay
        weight_decay = self.params['weight_decay']

        # Weights initialization
        weight_init = tf.keras.initializers.glorot_uniform(seed=self.seed)
        recurrent_init = tf.keras.initializers.orthogonal(seed=self.seed)

        # Model architecture
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Masking(mask_value = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0.], input_shape=(self.params['maxlen'], self.params['vocabulary_len'])))
        self.model.add(tf.keras.layers.Conv1D(self.params['conv_num_filter'], self.params['conv_kernel_size'], activation='relu',
                                              kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                              kernel_initializer=weight_init, 
                                              activity_regularizer=tf.keras.regularizers.l2(weight_decay)))
        self.model.add(tf.keras.layers.MaxPool1D())
        self.model.add(tf.keras.layers.Dropout(self.params['dropout_1_rate'], seed=self.seed))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.params['lstm_units'], return_sequences=False, 
                                                                         kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                                                         kernel_initializer=weight_init,
                                                                         recurrent_initializer=recurrent_init,
                                                                         recurrent_regularizer=tf.keras.regularizers.l2(weight_decay))))
        self.model.add(tf.keras.layers.Dropout(self.params['dropout_2_rate'], seed=self.seed))
        self.model.add(tf.keras.layers.Dense(10, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                            kernel_initializer=weight_init,
                                            activity_regularizer=tf.keras.regularizers.l2(weight_decay)))
        self.model.add(tf.keras.layers.Dropout(self.params['dropout_3_rate'], seed=self.seed))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid',
                                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                            kernel_initializer=weight_init,
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
        history = self.model.fit(x=X_tr, y=y_tr, epochs=epochs, shuffle=True, batch_size=self.batch_size,
                    callbacks=callbacks_list, validation_data=validation_data)
        trained_epochs = callbacks_list[0].stopped_epoch - callbacks_list[0].patience +1 if callbacks_list[0].stopped_epoch != 0 else epochs
        return history, trained_epochs

    def fit_early_stopping_by_loss_val(self, X_tr, y_tr, epochs, early_stopping_loss, callbacks_list, validation_data, shuffle=True):
        """
        Train model until current validation loss reaches holdout training loss specified by early_stopping_loss parameter. 
        
        Algorithm 7.3 (Ian Goodfellow, Yoshua Bengio, and Aaron Courville. 2016. Deep Learning. The MIT Press, pp. 246-250.)
        
        Params:
        -------
            :X_tr: training samples
            :y_tr: training labels
            :epochs: number of epochs training is performed on
            :early_stopping_loss: threshold loss - Once reached this loss the training is stopped
            :callbacks_list: list of callbacks to use in the training phase
            :validation_data: data to evaluate the model on at the end of each epoch
            :shuffle: if True, it shuffles data before starting the training
        
        """
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
        """
        It saves the model's weights into a hd5 file
        """
        with open(os.path.join(self.results_base_dir, "network_params"), 'wb') as params_pickle:
            pickle.dump(self.params, params_pickle)

        self.model.save_weights(os.path.join(self.results_base_dir, 'my_model_weights.h5'))
        model_json = self.model.to_json()
        with open(os.path.join(self.results_base_dir, "model.json"), "w") as json_file:
            json_file.write(model_json)    

    def fit_generator(self, generator, steps_per_epoch, epochs, validation_data=None, shuffle=True, callbacks_list=None):
        """
        Train the model for the same number of update step as in holdout validation phase
        
        Algorithm 7.2(Ian Goodfellow, Yoshua Bengio, and Aaron Courville. 2016. Deep Learning. The MIT Press, pp. 246-250.)
        """
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
                patience=5,
                monitor='val_loss',
                factor=0.75,
                verbose=1,
                min_lr=5e-6)
        ]
        return callbacks_list

    def predict(self,  x_test, batch_size: int = 32, verbose: int = 0) -> np.array:
        """
        Wrapper method for Keras model's method 'precict'

        Params:
        -------
            :x_test: test samples
            :batch_size: default=32
            :verbose: verbosity level
        """
        
        return self.model.predict(
            x_test,
            batch_size=batch_size,
            verbose=verbose,
            ).ravel()

    def predict_classes(self,  x_test, batch_size: int = 32, verbose: int = 1) -> np.array:
        """
        Wrapper method for Keras model's method 'precict_classes'

        Params:
        -------
            :x_test: test samples
            :batch_size: default=32
            :verbose: verbosity level

        Raise:
            Exception
        """
        try:
            return self.model.predict_classes(x_test)
        except Exception as err:
            print(f"EXCEPTION-RAISED: {err}")
            sys.exit(-1)
        pass
