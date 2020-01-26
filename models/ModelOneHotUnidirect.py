import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Masking
from metrics import f1_m, precision_m, recall_m

class ModelOneHotUnidirect():
    """
    This class defines the architecture used when the data are proteins, encoded
    by using one-hot encoding
    """

    def __init__(self, params):

        self.seed = params['seed']
        self.learning_rate = params['learning_rate']
        self.batch_size = params['batch_size']

        # It defines the initialization setup of weights
        weight_init = tf.keras.initializers.glorot_uniform(seed=self.seed)

        self.model = keras.Sequential(name="Unidirection-LSTM-Proteins-One_hot")
        self.model.add(Masking(mask_value = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0.], input_shape=(params['maxlen'], params['vocabulary_len'])))        
        self.model.add(keras.layers.LSTM(units=32, return_sequences = False,
                                    kernel_regularizer=keras.regularizers.l2(params['l2_regularizer']),
                                    kernel_initializer=weight_init                                                                                                          
                                    ))
        self.model.add(keras.layers.Dropout(0.2, seed=self.seed))
        self.model.add(keras.layers.Dense(units=1, activation='sigmoid', kernel_initializer=weight_init,
                                            kernel_regularizer=tf.keras.regularizers.l2(params['last_dense_l2_regularizer'])))
    

    def build(self, logger=None):
        """
        It compiles the model by defining optimizer, loss and learning rate
        """
        optimizer = tf.keras.optimizers.RMSprop(lr=lr, clipnorm=1.0)
        self.model.compile(loss='binary_crossentropy',
                            optimizer=optimizer,
                            metrics=['accuracy', f1_m, precision_m, recall_m])

        if logger is not None:
            self.model.summary(print_fn=lambda x: logger.info(x))
        else:
            self.model.summary()
        
    def fit(self, X_tr, y_tr, epochs, callback_list, validation_data, shuffle=True):
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
        history = self.model.fit(x=X_tr, y=y_tr, epochs=50, shuffle=True,
                    callbacks=callback_list, validation_data=validation_data)
        
        return history

    def fit_generator2(self, generator, steps_per_epoch, epochs, validation_data=None, shuffle=True, callbacks_list=None):
        history = self.model.fit_generator(generator, steps_per_epoch, epochs, shuffle=False, callbacks=self._get_callbacks(train=True),
                                           validation_data=validation_data)
        return history
    
    def evaluate():
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
        loss, accuracy, f1_score, precision, recall = model.evaluate(features, labels, verbose=0)
        return loss, accuracy, f1_score, precision, recall

    def _get_callbacks(self):
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
                filepath=os.path.join(self.results_base_dir, 'my_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            ),
            keras.callbacks.CSVLogger(os.path.join(self.results_base_dir, 'history.csv'))            
        ]
        return callbacks_list