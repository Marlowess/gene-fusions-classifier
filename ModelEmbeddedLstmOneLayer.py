# https://www.tensorflow.org/tutorials/text/text_generation
import tensorflow as tf
from tensorflow import keras

import copy

class ModelEmbeddedLstmOneLayer():

    def __init__(self, params: dict) -> None:
        self.params: dict = copy.deepcopy(params)
        self.model = None
        pass

    def build(self,) -> str:
        model = keras.Sequential()

        # NN Arch Params:
        input_dim: int = self.params['input_dim']
        output_dim: int = self.params['output_dim']
        mask_zero: int = self.params['mask_zero']

        lstm_units: int = self.params['lstm_units']

        seeds: list = self.params['seeds']
        l1_l2_reg: list = self.params['l1_l2_reg']
        dropout_rates: list = self.params['dropout_rates']

        # Compile Params:
        loss: str = self.params['loss']
        optimizer: str = self.params['optimizer']
        clip_norm: float = self.params['clip_norm']
        metrics: list = self.params['metrics']
        lr: float = self.params['lr']

        # Get a instance of a new sequential model
        model = keras.Sequential()

        # ---> Create layers
        # EMBEDDING
        emb_layer = tf.keras.layers.Embedding(
            # input_shape=input_shape,
            input_dim=input_dim,
            output_dim=output_dim,
            mask_zero=mask_zero,
            embeddings_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[0]),
            embeddings_regularizer=tf.keras.regularizers.L1L2(l1_l2_reg[0][0], l1_l2_reg[0][1]),
            # activity_regularizer=tf.keras.regularizers.L1L2(l1_l2_reg[1][0], l1_l2_reg[1][1]),
            name=f'embedding_layer_in{input_dim}_out{output_dim}')
    
        # DROPOUT #1
        dropout_layer_1 = tf.keras.layers.Dropout(dropout_rates[0])

        # LSTM #2
        lstm_layer = tf.keras.layers.LSTM(
            units=lstm_units,
            return_sequences=False,
            unit_forget_bias=True,

            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[1]),
            bias_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[2]),

            kernel_regularizer=tf.keras.regularizers.L1L2(l1_l2_reg[2][0], l1_l2_reg[2][1]),
            bias_regularizer=tf.keras.regularizers.L1L2(l1_l2_reg[3][0], l1_l2_reg[3][1]),
            # activity_regularizer=tf.keras.regularizers.L1L2(l1_l2_reg[4][0], l1_l2_reg[4][1]),
            name=f'lstm_1_units{lstm_units}')
    
        # DROPOUT #2
        dropout_layer_2 = tf.keras.layers.Dropout(dropout_rates[1])
    
        # DENSE #1
        dense_layer = tf.keras.layers.Dense(
            units=1,
            activation='sigmoid',

            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[3]),
            bias_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[4]),

            kernel_regularizer=tf.keras.regularizers.L1L2(l1_l2_reg[5][0], l1_l2_reg[5][1]),
            bias_regularizer=tf.keras.regularizers.L1L2(l1_l2_reg[6][0], l1_l2_reg[6][1]),
            # activity_regularizer=tf.keras.regularizers.L1L2(l1_l2_reg[7][0], l1_l2_reg[7][1]),
            name=f'dense_1_activation_sigmoid')
    

        # ---> Create NN stack
        model.add(emb_layer)
        model.add(dropout_layer_1)
        model.add(lstm_layer)
        model.add(dropout_layer_2)
        model.add(dense_layer)

        # ---> Compile the model
        optimizer_obj = self.get_optimizer(
            optimizer_name=optimizer.lower(),
            lr=lr,
            clipnorm=clip_norm)

        model.compile(loss=loss,
            optimizer=optimizer_obj,
            metrics=metrics)
        
        
        print(" [!] Display model summary")   
        print(model.summary())

        self.model = model
        summary_model_str: str = model.summary()

        return summary_model_str

    def get_optimizer(self, optimizer_name='adam', lr=0.001, clipnorm=1.0):
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
    
    def fit(self, X_tr, y_tr, epochs: int = 50, callback_list: list = None, validation_data = None, shuffle: bool = True) -> object:
        history = self.model.fit(x=X_tr, y=y_tr, epochs=epochs, shuffle=shuffle,
            callbacks=callback_list, validation_data=validation_data)
        return history
    
    def evaluate(self, X_test, y_test) -> object:
        scores = self.model.evaluate(X_test, y_test)
        print("{}: {}".format(self.model.metrics_names[1], scores[1] * 100))
        return scores
    
    def get_metrics_names(self,) -> object:
        return copy.deepcopy(self.model.metrics_names)
    pass