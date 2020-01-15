import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import auc as auc_test
from attlayer import AttentionWeightedAverage

class ModelEmbeddingBidirectProtein():

    def __init__(self, params):
        """
            This architecture is based on the one used for emoji NPL task
        """ 
        self.seed = 42

        # Architecture --- emoji network
        weight_init = tf.keras.initializers.glorot_uniform(seed=self.seed)

        # Variable-length int sequences.
        query_input = tf.keras.layers.Input(shape=(params['maxlen'],))

        # Masking layer
        masking_layer = tf.keras.layers.Masking(mask_value=0)(query_input)

        # Embedding lookup.  
        embed = tf.keras.layers.Embedding(params['vocabulary_len'], 
                                        params['embedding_size'], 
                                        embeddings_regularizer=tf.keras.regularizers.l2(params['embedding_regularizer']),
                                        embeddings_initializer=weight_init)(masking_layer)

        # Query embeddings of shape [batch_size, Tq, dimension].
        query_embeddings = tf.keras.layers.Activation('tanh')(embed)

        # Value embeddings of shape [batch_size, Tv, dimension].
        value_embeddings = tf.keras.layers.Activation('tanh')(embed)

        # Section A : embedding --> LSTM_1 --> LSTM_2
        lstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM((int)(params['embedding_size']/2), return_sequences=True,
                                                                    kernel_initializer=weight_init,
                                                                    recurrent_initializer=weight_init,
                                                                    kernel_regularizer=tf.keras.regularizers.l1_l2(params['l1_regularizer'], params['l2_regularizer'])
                                                                    ))(value_embeddings)
        dropout_lstm_1 = tf.keras.layers.Dropout(params['lstm_1_dropout_rate'], seed=self.seed)(lstm_1)

        lstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM((int)(params['embedding_size']/2), return_sequences=True,
                                                                    kernel_initializer=weight_init,
                                                                    recurrent_initializer=weight_init,
                                                                    kernel_regularizer=tf.keras.regularizers.l1_l2(params['l1_regularizer'], params['l2_regularizer'])
                                                                    ))(dropout_lstm_1)
        dropout_lstm_2 = tf.keras.layers.Dropout(params['lstm_2_dropout_rate'], seed=self.seed)(lstm_2)

        # Attention layer            
        concatenation = tf.keras.layers.concatenate([dropout_lstm_2, dropout_lstm_1, query_embeddings])
        
        attention_layer = AttentionWeightedAverage(name='attlayer', return_attention=False)(concatenation)

        # Attention dropout
        dropout_attention = tf.keras.layers.Dropout(params['attention_dropout'], seed=self.seed)(attention_layer)

        # Prediction layer
        prediction = tf.keras.layers.Dense(1, activation='sigmoid',
                                            kernel_initializer=weight_init,
                                            kernel_regularizer=tf.keras.regularizers.l2(params['last_dense_l2_regularizer']))(dropout_attention)

        # Build the model
        self.model = tf.keras.Model(inputs=[query_input],outputs=[prediction])
    

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
    
    def evaluate(self):
        print("evaluto")