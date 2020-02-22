from __future__ import absolute_import, division, print_function, unicode_literals

# Standard Imports.
import copy
import os
import sys
import pickle
import json
import pprint

# Main Machine Learning Imports.
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Custom Imports.
from utils.early_stopping_by_loss_val import EarlyStoppingByLossVal
from models.metrics import f1_m, precision_m, recall_m

class ModelUnidirect(object):

    def __init__(self, params:dict):
        
        # Create a own personal copy of input params for building model.
        self.params, pretrained_model_ = self._check_for_prentrained_model(params)
        pretrained_model = params.get("pretrained_model", None)

        self.weights_path = None

        # Get a new instance of a compiled model using tf functional API.
        # Check if the user wants a pre-trained model. If yes load the weights
        self.model = self._get_compiled_model(self.params, pretrained_model_)

        # Get a copy of callbacks in order to modify later such list
        # whether to validate or purely train this model
        # using either 7.2 or 7.2-enanched algorithms.
        self.callbacks = copy.deepcopy(self._getcallbacks(self.params))
        pass
    
    def _check_for_prentrained_model(self, _params: dict):
        params = copy.deepcopy(_params)
        if 'result_base_dir' in params.keys():
            results_base_dir = params['result_base_dir']
        else:
            results_base_dir = None

        if 'only_test' in params.keys():
            only_test = params['only_test']
        else:
            only_test = False

        if 'rnn_type' in params.keys():
            rnn_type = params['rnn_type']
        else:
            rnn_type = 'lstm'

        pretrained_model = params.get('pretrained_model', None)    
        if pretrained_model is not None:
            print("loading model")
            # train_dir = "/"
            # train_dir = train_dir.join(params['pretrained_model'].split("/")[:-1])                                              
            train_dir = params['pretrained_model']
            print(train_dir)
            params_path = os.path.join(train_dir, "network_params.pickle")
            print('params path:', params_path)
            with open(params_path, 'rb') as params_pickle:
                params = pickle.load(params_pickle)
                # sys.exit(0)
            params['result_base_dir'] = results_base_dir
            self.weights_path = os.path.join(train_dir, "my_model.h5")
            print('weights path:', self.weights_path)
            # sys.exit(0)
        params['only_test'] = only_test
        params['rnn_type'] = rnn_type

        pprint.pprint(params)
        return params, pretrained_model

    def _build_model(self, model_params: dict):

        # ----------------------------# 
        # Model's Params              #
        # ----------------------------#

        # Equal to timesteps
        sequence_len: int = model_params['maxlen']

        # Mapping a vocab-size to embbedding-size
        vocab_size: int = model_params['vocab_size']
        embedding_size: int = model_params['embedding_size']

        # RNN units
        rnns_units: list = model_params['rnns_units']
        rnn_type: str = model_params['rnn_type']

        # Regularization factors
        seeds: list = model_params['seeds']
        l1: list = model_params['l1']
        l2: list = model_params['l2']
        dropouts_rates: list = model_params['droputs_rates']
        recurrent_dropouts_rates: list = model_params['recurrent_dropouts_rates']

        onehot_flag: bool = model_params['onehot_flag']

        dense_units: list = model_params['dense_units']

        # ----------------------------# 
        # Build model                 #
        # ----------------------------#

        
        if onehot_flag is True:
            inputs = tf.keras.Input(shape=(sequence_len, vocab_size))
            x = tf.keras.layers.Masking(
                mask_value=[1.0, 0.0, 0.0, 0.0, 0.0],
                name="masking_layer")(inputs)
        else:
            inputs = tf.keras.Input(shape=(sequence_len,))
            x = tf.keras.layers.Masking(mask_value=0, name="masking_layer")(inputs)

            # Embedding layer
            x = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_size,
                # mask_zero=self.params['mask_zero'],
                embeddings_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[0]),
                embeddings_regularizer=tf.keras.regularizers.l2(l2[0]),
                name=f'embedding_layer_in{vocab_size}_out{embedding_size}')(x)

        # *** Block made from stack of LSTM layers
        index_reg = 1
        return_sequences: bool = True
        last_rnn_layer = len(rnns_units) - 1
        for index_lstm in range(len(rnns_units)):
            if index_lstm == last_rnn_layer:
                return_sequences = False
            x = self._get_rnn_layer(
                rnn_type,
                x,
                rnns_units,
                recurrent_dropouts_rates,
                l1,
                l2,
                dropouts_rates,
                seeds,
                index_lstm,
                index_reg,
                return_sequences=return_sequences
            )
        if return_sequences is True:
            x = tf.keras.layers.Flatten()(x)

        # *** Last nn block, classification block by means of dense layers.
        x = tf.keras.layers.Dense(
            units=dense_units[0],
            activation='relu',
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[2]),
            bias_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[2]),
            # kernel_regularizer=tf.keras.regularizers.l1(l1[2]),
            # kernel_regularizer=tf.keras.regularizers.l2(l2[2]),
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1[2], l2[2]),
        )(x)
        x = tf.keras.layers.Dropout(
            rate=dropouts_rates[2],
            seed=seeds[0]
        )(x)
        outputs = tf.keras.layers.Dense(
            units=1,
            activation='sigmoid',
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[3]),
            bias_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[3]),
            # kernel_regularizer=tf.keras.regularizers.l1(l1[3]),
            # kernel_regularizer=tf.keras.regularizers.l2(l2[3]),
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1[3], l2[3]),
        )(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='dna_genes_fusions_model')

        return model

    def _get_lstm_layer(self, x, rnns_units, recurrent_dropouts_rates, l1, l2, dropouts_rates, seeds, index_lstm, index_reg, return_sequences: bool = False):
        # tf.keras.layers.Bidirectional(
        x = \
            tf.keras.layers.LSTM(
                rnns_units[index_lstm],
                dropout=recurrent_dropouts_rates[index_lstm],
                # recurrent_dropout=recurrent_dropouts_rates[0],
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[index_reg]),
                recurrent_initializer=tf.keras.initializers.orthogonal(seed=seeds[index_reg]),
                # kernel_regularizer=tf.keras.regularizers.l1(l1[1]),
                # kernel_regularizer=tf.keras.regularizers.l2(l2[1]),
                return_sequences=return_sequences,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1[index_reg], l2[index_reg]),
            #    )
            )(x)
        # x = tf.keras.layers.Dropout(
        #     rate=dropouts_rates[index_reg],
        #     seed=seeds[index_reg]
        # )(x)
        return x

    def _get_rnn_layer(self, layer_type, x, rnns_units, recurrent_dropouts_rates, l1, l2, dropouts_rates, seeds, index_lstm, index_reg, return_sequences: bool = False):
        layer_type_: str= layer_type.lower()
        if layer_type_ == 'lstm':
            x = self._get_lstm_layer(
                x,
                rnns_units,
                recurrent_dropouts_rates,
                l1,
                l2,
                dropouts_rates,
                seeds,
                index_lstm,
                index_reg,
                return_sequences=return_sequences
            )
        elif layer_type_ == "gru":
            x = self._get_gru_layer(
                x,
                rnns_units,
                recurrent_dropouts_rates,
                l1,
                l2,
                dropouts_rates,
                seeds,
                index_lstm,
                index_reg,
                return_sequences=return_sequences
            )
        else:
            raise ValueError(f"ERROR: given rnn layer of type {layer_type_.upper()} is not allowed!")
        return x

    
    def _get_gru_layer(self, x, rnns_units, recurrent_dropouts_rates, l1, l2, dropouts_rates, seeds, index_lstm, index_reg, return_sequences: bool = False):
        # tf.keras.layers.Bidirectional(
        x = \
            tf.keras.layers.GRU(
                rnns_units[index_lstm],
                dropout=recurrent_dropouts_rates[index_lstm],
                # recurrent_dropout=recurrent_dropouts_rates[0],
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[index_reg]),
                recurrent_initializer=tf.keras.initializers.orthogonal(seed=seeds[index_reg]),
                # kernel_regularizer=tf.keras.regularizers.l1(l1[1]),
                # kernel_regularizer=tf.keras.regularizers.l2(l2[1]),
                return_sequences=return_sequences,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1[index_reg], l2[index_reg]),
            #    )
            )(x)
        # x = tf.keras.layers.Dropout(
        #     rate=dropouts_rates[index_reg],
        #     seed=seeds[index_reg]
        # )(x)
        return x
 
    def _get_conv_1d_layer(self, x, conv_filters, seeds, conv_kernel_size, conv_activations, l1, l2, dropouts_rates, index_conv, index_reg, last_layer):
        x = tf.keras.layers.Conv1D(
            filters=conv_filters[index_conv],
            strides=2,
            kernel_size=conv_kernel_size[index_conv],
            activation=conv_activations[index_conv],
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[index_reg]),
            bias_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[index_reg]),
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1[index_reg], l2[index_reg]),
        )(x)
        # x = tf.keras.layers.AveragePooling1D()(x)
        x = tf.keras.layers.MaxPool1D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.BatchNormalization()
        if last_layer == index_conv:
            return x
        x = tf.keras.layers.SpatialDropout1D(
            rate=dropouts_rates[index_reg],
            seed=seeds[index_reg]
        )(x)
        return x

    def _get_compiled_model(self, model_params: dict, pretrained_model):
        model = self._build_model(model_params)

        if pretrained_model is True:
            sys.exit(0)
            model.load_weights(self.weights_path)

        optimizer_name: str = model_params['optimizer']
        clip_norm: float = model_params['clip_norm']
        lr: float = model_params['lr']

        optimizer = self._get_optimizer(
            optimizer_name=optimizer_name.lower(),
            lr=lr,
            clipnorm=clip_norm)

        model.compile(loss='binary_crossentropy',
            # optimizer=optimizer_name.lower(),
            optimizer=optimizer,
            metrics=['accuracy', 'binary_crossentropy', f1_m, precision_m, recall_m])

        return model

    def _get_optimizer(self, optimizer_name='adam', lr=0.001, clipnorm=1.0):
        if optimizer_name == 'adadelta':
            optimizer = tf.keras.optimizers.Adadelta(lr, clipnorm=clipnorm)
        if optimizer_name == 'adagrad':
            optimizer = tf.keras.optimizers.Adagrad(lr, clipnorm=clipnorm)
        elif optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr, clipnorm=clipnorm)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(lr, clipnorm=clipnorm)
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr, clipnorm=clipnorm)
        elif optimizer_name == 'nadam':
            optimizer = tf.keras.optimizers.Nadam(lr, clipnorm=clipnorm)
        else:
            raise ValueError(f'ERROR: specified optimizer name {optimizer_name} is not allowed.')
        return optimizer

    def _getcallbacks(self, model_params) -> list:

        if model_params['only_test'] is True:
            return None

        train_result_path: str = model_params['result_base_dir']
        callbacks_list = [            
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                min_delta=0,
                restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(train_result_path, 'my_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            ),
            keras.callbacks.CSVLogger(os.path.join(train_result_path, 'history.csv')),
            keras.callbacks.ReduceLROnPlateau(
                patience=5,
                monitor='val_loss',
                factor=0.75,
                verbose=1,
                min_lr=5e-6)
        ]
        return callbacks_list

    def evaluate(self, x_test, y_test) -> dict:
        scores = self.model.evaluate(x_test, y_test, verbose=0)
        results_dict = dict(zip(self.model.metrics_names, scores))
        return results_dict

    def build(self, logger):
        if logger is not None:
            summary_list: list = list()
            # self.model.summary(print_fn=lambda x: logger.info(x))
            self.model.summary(print_fn=lambda x: summary_list.append(str(x)) )
            logger.info('\n' + '\n'.join([str(xi) for xi in summary_list]))
        else:
            self.model.summary()
        pass
    
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
        
        assert self.model != None

        print('x_train shape', X_tr.shape)
        print('y_train shape', y_tr.shape)

        callbacks_list = self.callbacks
        batch_size = self.params['batch_size']
        
        history = self.model.fit(x=X_tr, y=y_tr, epochs=epochs, shuffle=True, batch_size=batch_size,
                    callbacks=callbacks_list, validation_data=validation_data)
        trained_epochs = callbacks_list[0].stopped_epoch - callbacks_list[0].patience +1 if callbacks_list[0].stopped_epoch != 0 else epochs
        return history, trained_epochs

    def fit_generator2(self, generator, steps_per_epoch, epochs, validation_data=None, shuffle=True, callbacks_list=None):

        assert self.model != None

        # Remove early stopping
        # assert self.callbacks != None
        callbacks_copy: list = copy.deepcopy(self.callbacks)
        # assert callbacks_copy != None

        print(callbacks_copy[1:])
        callbacks_copy = callbacks_copy[1:]

        history = self.model.fit_generator(
            generator,
            steps_per_epoch,
            epochs,
            shuffle=False,
            callbacks=callbacks_copy, # Since we have removed Early-Stopping
            validation_data=validation_data)
        return history

    def fit_early_stopping_by_loss_val(self, X_tr, y_tr, epochs, early_stopping_loss, callbacks_list, validation_data, shuffle=True):
        
        assert self.model != None

        print(f"early stopping loss{early_stopping_loss}")
        callbacks_list = copy.deepcopy(self.callbacks)
        callbacks_list.append(EarlyStoppingByLossVal(monitor='val_loss', value=early_stopping_loss))
        history = self.model.fit(x=X_tr, y=y_tr, epochs=epochs, shuffle=True,
                    callbacks=callbacks_list, validation_data=validation_data)
        
        return history
    
    def train(self, x_train, y_train, epochs: int = 10, batch_size: int = 32, shuffle: bool = True, validation_data: tuple = None):
        
        assert self.model != None

        print('x_train shape', x_train.shape)
        print('y_train shape', y_train.shape)

        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            validation_data=validation_data,
            shuffle=shuffle,
            # callbacks=self.callbacks
        )
        return history

    def save_weights(self):

        assert self.model != None

        results_base_dir: str = self.params['result_base_dir']
        with open(os.path.join(results_base_dir, "network_params.pickle"), 'wb') as params_pickle:
            pickle.dump(self.params, params_pickle)

        self.model.save_weights(os.path.join(f"{results_base_dir}", 'my_model_weights.h5'))
        model_json = self.model.to_json()
        with open(os.path.join(f"{results_base_dir}", "model.json"), "w") as json_file:
            json_file.write(model_json)

    def predict(self,  x_test, batch_size: int = 32, verbose: int = 0) -> np.array:
        # return np.asarray([])
        
        assert self.model != None

        return self.model.predict(
            x_test,
            batch_size=batch_size,
            verbose=verbose,
            ).ravel()

    def predict_classes(self,  x_test, batch_size: int = 32, verbose: int = 1) -> np.array:
        # return np.asarray([])

        assert self.model != None

        try:
            type_api: str = self.params['meta_info']['api']
            if type_api == 'functional':
                return self._predict_classes_funcitonal_api(
                    x=x_test,
                    # batch_size=batch_size,
                    # verbose=verbose
                )
            elif type_api == 'sequential':
                return self.model.predict_classes(
                    x_test,
                )
            else:
                raise ValueError(f"ERROR: type api {type_api} not allowed.")
        except Exception as err:
            print(f"EXCEPTION-RAISED: {err}")
            sys.exit(-1)
        pass

    def _predict_classes_funcitonal_api(self, x, batch_size=32, verbose=1):
        '''Generate class predictions for the input samples
        batch by batch.
        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.
        # Returns
            A numpy array of class predictions.
        '''

        """
        if proba.shape[-1] > 1:
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')
        """

        assert self.model != None

        proba = self.predict(x, batch_size=batch_size, verbose=verbose)
        return np.asarray(list(map(lambda xi: 1 if xi >= 0.50 else 0, proba)), dtype=np.int)

    def save_model(self):

        assert self.model != None

        tf.keras.models.save_model(
            model=self.model,
            filepath=self.params["model_path"],
        )
        pass  

    def plot_model(self,) -> None:

        assert self.model != None

        full_model_image_path: str = os.path.join('', 'model_graph.png')
        tf.keras.utils.plot_model(self.model, full_model_image_path, show_shapes=True)
    pass