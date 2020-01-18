import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Masking
from metrics import f1_m, precision_m, recall_m

class ModelOneHotProtein():
    """
    This class defines the architecture used when the data are proteins, encoded
    by using one-hot encoding
    """

    def __init__(self, params):
        self.params = params
        self.seed = 42
        self.learning_rate = params['learning_rate']
        print(self.learning_rate)
        self.batch_size = params['batch_size']

        # It defines the initialization setup of weights

        self.model = keras.Sequential(name="Unidirection-LSTM-Proteins-One_hot")
        self.model.add(Masking(mask_value = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0.], input_shape=(params['maxlen'], params['vocabulary_len'])))        
        self.model.add(keras.layers.LSTM(units=32, return_sequences = False,
                                    kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01),
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0)                                                                                                          
                                    ))
        self.model.add(keras.layers.Dropout(0.2, seed=10))
        # self.model.add(keras.layers.Dense(units=1, activation='sigmoid',
        #                                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=13),
        #                                   kernel_regularizer=tf.keras.regularizers.l2(params['last_dense_l2_regularizer'])))
        self.model.add(keras.layers.Dense(units=1, activation='sigmoid', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=17)))
    

    def build(self, logger):
        """
        It compiles the model by defining optimizer, loss and learning rate
        """
        optimizer = tf.keras.optimizers.RMSprop(lr=self.params['learning_rate'], clipnorm=1.0)
        self.model.compile(loss='binary_crossentropy',
                            optimizer=optimizer,
                            metrics=['accuracy'])#, f1_m, precision_m, recall_m])

        self.model.summary(print_fn=lambda x:logger.info(x))
        
    def fit(self, X_tr, y_tr, epochs, callbacks_list, validation_data, shuffle=True):
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
        history = self.model.fit(x=X_tr, y=y_tr, epochs=epochs, shuffle=True,
                    callbacks=callbacks_list, validation_data=validation_data)
        trained_epochs = callbacks_list[0].stopped_epoch if callbacks_list[0].stopped_epoch != 0 else epochs
        
        return history, trained_epochs
    
    def fit_generator(self, generator, steps, validation_data=None, shuffle=True, callbacks_list=None):
        history = self.model.fit_generator(generator, steps, shuffle=True, callbacks=callbacks_list,
                                           validation_data=validation_data)
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
        scores = self.model.evaluate(X, y, verbose=0)
        metrics = dict(zip(self.model.metrics_names, scores))

        return metrics
    
    def plot_model(self) -> None:
        tf.keras.utils.plot_model(self.model, 'model_graph.png', show_shapes=True)