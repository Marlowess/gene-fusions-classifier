import tensorflow as tf
import logging
from utils import print_and_write
import os

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
    def __init__(self, n_vocab, log_dir=".", model_name="",
                 hidden_size=128, 
                 lr=0.001, loss='binary_crossentropy',
                 seed=42):
        
        _init_logger(log_dir)
        weight_init = tf.keras.initializers.glorot_uniform(seed=seed)
        optimizer = tf.keras.optimizers.RMSprop(lr=lr)
        
        self.model = tf.keras.Sequential(name=model_name)
        self.model.add(tf.keras.layers.Embedding(n_vocab, hidden_size))
        self.model.add(tf.keras.layers.LSTM(64, dropout=0.2,recurrent_dropout=0.4))

    #     model.add(tf.keras.layers.CuDNNLSTM(units=n_units,
    #                                   name='LSTM_layer',
    #                                   return_sequences=False,
    #                                   kernel_initializer=weight_init))               
        
        self.model.add(tf.keras.layers.Dense(1, name='Dense',
                            activation='sigmoid',
                            kernel_initializer=weight_init))

        # Build the optimizer
        # self.build_optimizer(learning_rate, optimizer)
        # Compile model
        self.model.compile(loss=loss, 
                        optimizer=optimizer,
                        metrics=['accuracy'])

        # Print the netowrk's architecture
        self.model.summary(print_fn=lambda x: logger.info(x))
        # Print the network's configurations
        # log_file.write(model.summary()) 
        logger.info("Loss: {}\n".format(loss))
        logger.info("Optimizer: {}\n\n".format(optimizer))
