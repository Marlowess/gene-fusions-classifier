# import tensorflow as tf
import logging
import keras
from utils import print_and_write
import os
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention

logger= logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def _init_logger(log_dir):
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
    # file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'model.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    # stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    # add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

class Model:
    def __init__(self, n_vocab=None, log_dir=".", model_name="", hidden_size=128,
                 lr=0.001, lstm_units=64, loss='binary_crossentropy', dropout_rate=0.0,
                 recurrent_dropout_rate=0.0, load=False, seed=42):

        if (load == False):
            self._new_model(n_vocab, log_dir, model_name, hidden_size, lstm_units, lr,
                        loss, dropout_rate, recurrent_dropout_rate, seed)
        # Todo refactor if load model call function else put directly code for new model instead of repeating parameters
        else:
            self._load_model(log_dir)
            
    def _new_model(self, n_vocab, log_dir=".", model_name="",
                 hidden_size=128, lstm_units=64, lr=0.001, loss='binary_crossentropy',
                 dropout_rate=0.0, recurrent_dropout_rate=0.0, seed=42):
        _init_logger(log_dir)
        weight_init = keras.initializers.glorot_uniform(seed=seed)
        optimizer = keras.optimizers.RMSprop(lr=lr, clipnorm=1.0)
        
        # self.model = keras.models.Sequential(name=model_name)
        inputs = keras.layers.Input(batch_shape=(None, None), dtype='int32')
        print(inputs.shape)
        l2_reg = 0.06
        # l2_reg = 0
        logger.info('l2 reg: {}'.format(l2_reg))
        out1 = keras.layers.Embedding(n_vocab, hidden_size, embeddings_regularizer=keras.regularizers.l2(l2_reg)
                                      , mask_zero=True)(inputs)
        print(out1.shape)
        out2 = keras.layers.Bidirectional(keras.layers.LSTM(units=64, dropout=dropout_rate,recurrent_dropout=recurrent_dropout_rate,
                                            return_sequences=True))(out1)
        out3 = keras.layers.Bidirectional(keras.layers.LSTM(units=128, dropout=dropout_rate,recurrent_dropout=recurrent_dropout_rate,
                                            return_sequences=True))(out2)
        out4 = keras.layers.merge.Concatenate()([out3, out2, out1])
        out5 = SeqWeightedAttention()(out4)
        out6 = keras.layers.Dropout(0.5)(out5)
        # self.model.add(keras.layers.Flatten())
        predictions = keras.layers.Dense(units=1, name='Dense',
                            activation='sigmoid', kernel_initializer=weight_init) (out5)

        self.model = keras.models.Model(inputs=inputs, outputs=predictions)
        self.model.compile(loss=loss, 
                        optimizer=optimizer,
                        metrics=['accuracy'])
        

        # self.model.add(keras.layers.Embedding(n_vocab, hidden_size, mask_zero=True))
        # self.model.add(keras.layers.LSTM(64, dropout=dropout_rate,recurrent_dropout=recurrent_dropout_rate,
                                            # return_sequences=True))
        # self.model.add(keras.layers.LSTM(64, return_sequences=True))

    #     model.add(tf.keras.layers.CuDNNLSTM(units=n_units,
    #                                   name='LSTM_layer',
    #                                   return_sequences=False,
    #                                   kernel_initializer=weight_init))               
        # self.model.add(SeqWeightedAttention())
        # self.model.add(keras.layers.Flatten())
        # self.model.add(keras.layers.Dense(units=1, name='Dense',
                            # activation='sigmoid',
                            # kernel_initializer=weight_init))

        # Build the optimizer
        # self.build_optimizer(learning_rate, optimizer)
        # Compile model
        # self.model.compile(loss=loss, 
        #                 optimizer=optimizer,
        #                 metrics=['accuracy'])

        # Print the netowrk's architecture
        self.model.summary(print_fn=lambda x: logger.info(x))
        # Print the network's configurations
        # log_file.write(model.summary()) 
        logger.info("Loss: {}\n".format(loss))
        logger.info("Optimizer: {}\n\n".format(optimizer))

    def _load_model(self, log_dir):
        self.model = keras.models.load_model(os.path.join(log_dir, 'model.hdf5'))