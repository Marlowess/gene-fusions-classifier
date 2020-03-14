import tensorflow as tf
from tensorflow import keras
import os
import pickle
import numpy as np
from tensorflow.keras.layers import Masking
from models.metrics import f1_m, precision_m, recall_m
from utils.early_stopping_by_loss_val import EarlyStoppingByLossVal
import json

from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Embedding

class ModelKMers():

    def __init__(self,):
        self.model = None

        self.learning_rate: float = None
        self.batch_size: int = None

        self.results_base_dir: str = None
        self.params: dict = None

        
        self.vocab_size = self.params['']

        model = keras.Sequential()

        history_filename = "le1-3history.txt"

        model.add(tf.keras.Input(shape=(15000,)))
        # model.add(Masking(mask_value=0, name="masking_layer"))
        model.add(Embedding(input_dim=self.vocab_size, output_dim=16, mask_zero=True))
        # model.add(Embedding(1000, 64, input_length=10))

        # self.model.add(keras.layers.LSTM(units=self.params['lstm1']['units'], return_sequences = True,
                                #  dropout=self.params['lstm1']['dropout'],
                                #  kernel_regularizer=keras.regularizers.l1l2(l1=0.0001, l2=0.001)
                                #  recurrent_regularizer=keras.regularizers.l2(l=self.params['lstm1']['recurrent_l2']),
                                #  activity_regularizer=keras.regularizers.l2(l=self.params['lstm1']['activation_l2'])
                                #  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=1) 
                                #  ))
        model.add(tf.keras.layers.Bidirectional(keras.layers.LSTM(units=16, return_sequences = False,
                                  dropout=0.3,
                                #  kernel_regularizer=keras.regularizers.l2(l=self.params['lstm2']['kernel_l2']),
                                  kernel_regularizer=keras.regularizers.l2(0.01),
                                  recurrent_regularizer=keras.regularizers.l2(0.01)
                                  # recurrent_regularizer=keras.regularizers.l2(l=self.params['lstm2']['recurrent_l2'])
                                  )))
        # self.model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(rate=0.3))
        model.add(keras.layers.Dense(units=32, activation='relu',
                                  kernel_regularizer=keras.regularizers.l2(0.01)))
                                #   kernel_regularizer=keras.regularizers.l2(self.params['dense1']['kernel_l2'])))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(units=1, activation='sigmoid',
                                  kernel_regularizer=keras.regularizers.l2(0.01)))
                                #   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=17)
                                # ))

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
                     verbose=1
                 ),
                  keras.callbacks.CSVLogger(history_filename),
                  keras.callbacks.ReduceLROnPlateau(
                      patience=5,
                      monitor='val_loss',
                      factor=0.75,
                      verbose=1,
                      min_lr=5e-6)
        ]

        self.model = model

        pass

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