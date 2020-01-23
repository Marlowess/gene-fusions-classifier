# https://www.tensorflow.org/tutorials/text/text_generation
import tensorflow as tf
from tensorflow import keras
import os
from models.attlayer import AttentionWeightedAverage
from models.metrics import f1_m, precision_m, recall_m

import copy

class ModelEmbeddingUnidirect():
    """
    Unidirectional LSTM network 
    """
    def __init__(self, params: dict) -> None:
        # Network's parameter, read from file
        self.params: dict = copy.deepcopy(params)        
        
        # Initialize the keras sequencial model
        self.model = keras.Sequential()

        self.model.add(tf.keras.layers.Input(shape=(self.params['maxlen'],)))

        self.model.add(tf.keras.layers.Masking(mask_value=0, name="masking_layer"))

        # Embedding layer
        self.model.add(tf.keras.layers.Embedding(
            input_dim=self.params['vocab_size'],
            output_dim=self.params['embedding_size'],
            # mask_zero=self.params['mask_zero'],
            embeddings_initializer=tf.keras.initializers.glorot_uniform(seed=self.params['seeds'][0]),
            embeddings_regularizer=tf.keras.regularizers.l2(self.params['l2_regularizer']),
            name=f'embedding_layer_in{self.params["vocab_size"]}_out{self.params["embedding_size"]}'))
        
        # Dropout after the embedding layer
        self.model.add(tf.keras.layers.Dropout(self.params['embedding_dropout_rate'], seed=self.params['seeds'][0]))

        # First LSTM layer
        self.model.add(tf.keras.layers.LSTM(
            units=self.params['lstm_units'],
            return_sequences=False,
            unit_forget_bias=True,
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.params['seeds'][1]),
            bias_initializer=tf.keras.initializers.glorot_uniform(seed=self.params['seeds'][2]),
            kernel_regularizer=tf.keras.regularizers.l2(self.params['l2_regularizer']),
            name=f'lstm_1_units{self.params["lstm_units"]}'))  

        # Dropout after the lstm layer
        self.model.add(tf.keras.layers.Dropout(self.params['lstm_output_dropout'], seed=self.params['seeds'][0]))

        # Fully connected (prediction) layer
        self.model.add(tf.keras.layers.Dense(
            units=1,
            activation='sigmoid',
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.params['seeds'][3]),
            bias_initializer=tf.keras.initializers.glorot_uniform(seed=self.params['seeds'][4]),
            kernel_regularizer=tf.keras.regularizers.l2(self.params['last_dense_l2_regularizer']),
            name=f'dense_1_activation_sigmoid'))

    def build(self, logger=None) -> str:        
        optimizer: str = self.params['optimizer']
        clip_norm: float = self.params['clip_norm']
        lr: float = self.params['lr']
        
        optimizer_obj = self._get_optimizer(
            optimizer_name=optimizer.lower(),
            lr=lr,
            clipnorm=clip_norm)

        self.model.compile(loss='binary_crossentropy',
                            optimizer=optimizer_obj,
                            metrics=['accuracy', 'binary_crossentropy', f1_m, precision_m, recall_m])
        
        summary_str: str = ''
        # self.model.summary(print_fn=lambda x: str(summary_str + '\n' + str(x)))
        return '\n' + summary_str

    def _get_optimizer(self, optimizer_name='adam', lr=0.001, clipnorm=1.0):
        if optimizer_name == 'adadelta':
            optimizer = tf.keras.optimizers.Adadelta(lr, clipnorm=clipnorm)
        elif optimizer_name == 'adagrad':
            optimizer = tf.keras.optimizers.Adagrad(lr, clipnorm=clipnorm)
        elif optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr, clipnorm=clipnorm)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(lr, clipnorm=clipnorm)
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr, clipnorm=clipnorm)
        elif optimizer_name == 'nadam':
            optimizer = tf.keras.optimizers.Nadam(lr, clipnorm=clipnorm)
        return optimizer
    
    def fit(self, X_tr, y_tr, epochs: int = 50, callbacks_list: list = None, validation_data = None, shuffle: bool = True) -> object:
        history = self.model.fit(
            x=X_tr,
            y=y_tr,
            epochs=epochs,
            shuffle=shuffle,
            callbacks=self._get_callbacks(),
            validation_data=validation_data)
        # trained_epochs = callbacks_list[0].stopped_epoch - callbacks_list[0].patience +1 if callbacks_list[0].stopped_epoch != 0 else epochs
        return history, 0 # trained_epochs
    
    def evaluate(self, X_test, y_test) -> dict:
        scores = self.model.evaluate(X_test, y_test)
        # print("{}: {}".format(self.model.metrics_names[1], scores[1] * 100))

        result_scores_dict: dict = dict(zip(self.get_metrics_names(), scores))
        return result_scores_dict
    
    def get_metrics_names(self,) -> object:
        return copy.deepcopy(self.model.metrics_names)
    
    def plot_model(self,) -> None:
        tf.keras.utils.plot_model(self.model, 'model_graph.png', show_shapes=True)

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
                filepath=os.path.join(self.params['result_base_dir'], 'my_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            ),
            keras.callbacks.CSVLogger(os.path.join(self.params['result_base_dir'], 'history.csv')),
            keras.callbacks.ReduceLROnPlateau(
                patience=10,
                monitor='val_loss',
                factor=0.75,
                verbose=1,
                min_lr=5e-6)
        ]
        return callbacks_list