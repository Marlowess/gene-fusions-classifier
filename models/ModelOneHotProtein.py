import tensorflow as tf
import os
import pickle
from tensorflow import keras
from tensorflow.keras.layers import Masking
from models.metrics import f1_m, precision_m, recall_m
from utils.early_stopping_by_loss_val import EarlyStoppingByLossVal
import json

class ModelOneHotProtein():
    """
    This class defines the architecture used when the data are proteins, encoded
    by using one-hot encoding
    """

    def __init__(self, params):

        self.results_base_dir = params['result_base_dir']

        # if pretrained model is defined i overwrite params with the ones of loaded model
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

        self.batch_size = self.params['batch_size']

        # It defines the initialization setup of weights

        self.model = keras.Sequential(name="Unidirection-LSTM-Proteins-One_hot")
        self.model.add(Masking(mask_value = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0.], input_shape=(self.params['maxlen'], self.params['vocabulary_len'])))        
        self.model.add(keras.layers.LSTM(units=self.params['lstm1']['units'], return_sequences = True,
                                         dropout=self.params['lstm1']['dropout'],
                                         kernel_regularizer=keras.regularizers.l2(l=self.params['lstm1']['kernel_l2']),
                                         recurrent_regularizer=keras.regularizers.l2(l=self.params['lstm1']['recurrent_l2']),
                                        #  activity_regularizer=keras.regularizers.l2(l=self.params['lstm1']['activation_l2'])
                                        #  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=1) 
                                         ))
        self.model.add(keras.layers.LSTM(units=self.params['lstm2']['units'], return_sequences = False,
                                         dropout=self.params['lstm2']['dropout'],
                                         kernel_regularizer=keras.regularizers.l2(l=self.params['lstm2']['kernel_l2']),
                                         recurrent_regularizer=keras.regularizers.l2(l=self.params['lstm2']['recurrent_l2']),
                                        #  activity_regularizer=keras.regularizers.l2(l=self.params['lstm2']['activation_l2'])
        #                                 #  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=2)
                                         ))
        # self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dropout(rate=self.params['dense1']['dropout']))
        self.model.add(keras.layers.Dense(units=self.params['dense1']['units'], activation='relu',
                                          kernel_regularizer=keras.regularizers.l2(self.params['dense1']['kernel_l2'])))
        self.model.add(keras.layers.Dropout(self.params['dense2']['dropout']))
        self.model.add(keras.layers.Dense(units=1, activation='sigmoid',
                                        #   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=17)
                                        ))
    
        # Check if the user wants a pre-trained model. If yes load the weights
        if self.pretrained_model is not None:
            self.model.load_weights(self.pretrained_model)

    def build(self, logger):
        """
        It compiles the model by defining optimizer, loss and learning rate
        """
        optimizer = tf.keras.optimizers.RMSprop(lr=self.params['lr'], clipnorm=1.0)
        self.model.compile(loss='binary_crossentropy',
                            optimizer=optimizer,
                            metrics=['accuracy', f1_m, precision_m, recall_m])#, f1_m, precision_m, recall_m])

        self.model.summary(print_fn=lambda x:logger.info(x))
        
        if logger is not None:
            logger.info("\n" + json.dumps(self.params, indent=4))
         
    def fit(self, X_tr, y_tr, epochs, callbacks_list, validation_data, shuffle=True, early_stopping_loss=False):
        """
        Fit the model with the provided data and returns the results
        Inputs:
        - X_tr: samples
        - y_tr: labels related to the samples
        - epochs: number of epochs before stopping the training
        - callback_list
        - validation_data: data the model is validated on each time a epoch is completed
        - shuffle: if the dataset has to be shuffled before being fed into the network

        Outputs:
        - history: it contains the results of the training
        """
        callbacks_list = self._get_callbacks()
        history = self.model.fit(x=X_tr, y=y_tr, epochs=self.params['epochs'], shuffle=True,
                    callbacks=callbacks_list, validation_data=validation_data)
        trained_epochs = callbacks_list[0].stopped_epoch - callbacks_list[0].patience +1 \
            if callbacks_list[0].stopped_epoch != 0 else self.params['epochs']
        
        return history, trained_epochs
    
    def fit_generator2(self, generator, steps_per_epoch, epochs, validation_data=None, shuffle=True, callbacks_list=None):
        history = self.model.fit_generator(generator, steps_per_epoch, epochs, shuffle=False, callbacks=self._get_callbacks(train=True),
                                           validation_data=validation_data)
        return history
    
    def fit_early_stopping_by_loss_val(self, X_tr, y_tr, epochs, early_stopping_loss, callbacks_list, validation_data, shuffle=True):
        print(f"early stopping loss: {early_stopping_loss}")
        callbacks_list = self._get_callbacks(train=True)
        callbacks_list.append(EarlyStoppingByLossVal(monitor='val_loss', value=early_stopping_loss))
        history = self.model.fit(x=X_tr, y=y_tr, epochs=epochs, shuffle=True,
                    callbacks=callbacks_list, validation_data=validation_data)
        
        return history
    
    def evaluate(self, X, y):
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
        scores = self.model.evaluate(X, y, verbose=1)
        metrics = dict(zip(self.model.metrics_names, scores))

        return metrics
    
    def plot_model(self) -> None:
        tf.keras.utils.plot_model(self.model, 'model_graph.png', show_shapes=True)

    def _get_callbacks(self, train=False):
        """
        It defines the callbacks for this specific architecture
        """
        if (not train):
            callbacks_list = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.params['early_stopping_patiente'],
                    restore_best_weights=True
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.results_base_dir, 'my_model.h5'),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=0
                ),
                keras.callbacks.CSVLogger(os.path.join(self.results_base_dir, 'history.csv')),
                # keras.callbacks.ReduceLROnPlateau(
                #     patience=10,
                #     monitor='val_loss',
                #     factor=0.75,
                #     verbose=1,
                #     min_lr=5e-6)
            ]

        else:
            callbacks_list = [
                keras.callbacks.CSVLogger(os.path.join(self.results_base_dir, 'history.csv'))
            ]

        return callbacks_list

    def save_weights(self):
        with open(os.path.join(self.results_base_dir, "network_params"), 'wb') as params_pickle:
            pickle.dump(self.params, params_pickle)

        self.model.save_weights(os.path.join(self.results_base_dir, 'my_model_weights.h5'))
        model_json = self.model.to_json()
        with open(os.path.join(self.results_base_dir, "model.json"), "w") as json_file:
            json_file.write(model_json)
            