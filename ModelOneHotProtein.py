import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Masking

class ModelOneHotProtein():

    def __init__(self, params):
        self.model = keras.Sequential()

        self.model.add(Masking(mask_value = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0.], input_shape=(params['maxlen'], params['vocabulary_len'])))
        # model.add(keras.layers.LSTM(units=128, return_sequences = False))
        # model.add(keras.layers.Dropout(0.2))

        self.model.add(keras.layers.LSTM(units=32, return_sequences = False,
                                    kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                                        #  kernel_initializer=glorot_uniform(seed=0),
                                        #  input_shape=(X_tr_masked.shape[1], X_tr_masked.shape[2]
                                    ))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    

    def build(self, lr=5e-4):
        optimizer = tf.keras.optimizers.RMSprop(lr=lr, clipnorm=1.0)

        self.model.compile(loss='binary_crossentropy',
                            optimizer=optimizer,
                            metrics=['accuracy'])

        self.model.summary()
        
    def fit(self, X_tr, y_tr, epochs, callback_list, validation_data, shuffle=True):
        history = self.model.fit(x=X_tr, y=y_tr, epochs=50, shuffle=True,
                    callbacks=callback_list, validation_data=validation_data)
        
        return history
    
    def evaluate():
        print("evaluto")