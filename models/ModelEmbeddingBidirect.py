import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import auc as auc_test
from models.attlayer import AttentionWeightedAverage
from models.metrics import f1_m, precision_m, recall_m

class ModelEmbeddingBidirect():
    """
    This architecture is based on https://arxiv.org/abs/1708.00524
    Two bidirectional LSTM layers and one attention layer in order to focus the attention
    only onto the important parts of the fusion
    """ 

    def __init__(self, params):
        """
        It initializes the model before the training
        """

        self.seed = 42
        self.learning_rate = params['learning_rate']
        self.batch_size = params['batch_size']                  

        # defines where to save the model's checkpoints 
        self.results_base_dir = params['result_base_dir']  

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
        lstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM((int)(params['embedding_size']/2), return_sequences=True,
                                                                    kernel_initializer=weight_init,
                                                                    recurrent_initializer=weight_init,
                                                                    kernel_regularizer=tf.keras.regularizers.l1_l2(params['l1_regularizer'], params['l2_regularizer'])
                                                                     ))(value_embeddings)

        # Dropout layer after the first LSTM layer                                                                        
        dropout_lstm_1 = tf.keras.layers.Dropout(params['lstm_1_dropout_rate'], seed=self.seed)(lstm_1)

        # Second LSTM layer
        lstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM((int)(params['embedding_size']/2), return_sequences=True,
                                                                    kernel_initializer=weight_init,
                                                                    recurrent_initializer=weight_init,
                                                                    kernel_regularizer=tf.keras.regularizers.l1_l2(params['l1_regularizer'], params['l2_regularizer'])
                                                                    ))(dropout_lstm_1)

        # Dropout layer after the second LSTM layer                                                                    
        dropout_lstm_2 = tf.keras.layers.Dropout(params['lstm_2_dropout_rate'], seed=self.seed)(lstm_2)

        # Data combination before the attention layer            
        concatenation = tf.keras.layers.concatenate([dropout_lstm_2, dropout_lstm_1, query_embeddings])
        
        # Attention layer: the implementation can be found at 
        # https://github.com/bfelbo/DeepMoji/blob/master/deepmoji/model_def.py
        attention_layer = AttentionWeightedAverage(name='attlayer', return_attention=False)(concatenation)

        # Attention dropout
        dropout_attention = tf.keras.layers.Dropout(params['attention_dropout'], seed=self.seed)(attention_layer)

        # Prediction layer
        prediction = tf.keras.layers.Dense(1, activation='sigmoid',
                                            kernel_initializer=weight_init,
                                            kernel_regularizer=tf.keras.regularizers.l2(params['last_dense_l2_regularizer']))(dropout_attention)

        # Build the model
        self.model = tf.keras.Model(inputs=[query_input],outputs=[prediction])

        # Check if the user wants a pre-trained model. If yes load the weights
        if params['pretrained_model'] is not None:
            self.model.load_weights(params['pretrained_model'])
    

    def build(self, logger=None):
        """
        It compiles the model by defining optimizer, loss and learning rate
        """
        optimizer = tf.keras.optimizers.RMSprop(lr=self.learning_rate, clipnorm=1.0)
        self.model.compile(loss='binary_crossentropy',
                            optimizer=optimizer,
                            metrics=['accuracy', f1_m, precision_m, recall_m])
        if (logger is not None):
            self.model.summary(print_fn=lambda x: logger.info(x))
        else:
            self.model.summary()
            
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
        history = self.model.fit(x=X_tr, y=y_tr, epochs=epochs, shuffle=True, batch_size=self.batch_size,
                    callbacks=_get_callbacks(), validation_data=validation_data)
        return history
    
    def evaluate(self, features, labels):
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
        loss, accuracy, f1_score, precision, recall = self.model.evaluate(features, labels, verbose=0)
        metrics_value = [loss, accuracy, f1_score, precision, recall]

        results_dict = dict(zip(self.model.metrics, metrics_value))
        return results_dict

    def print_metric(name, value):
        print('{}: {}'.format(name, value))

    def save_weights(self):
        pass    

    def fit_generator(self):
        pass    
    
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
                filepath=os.path.join(self.results_base_dir, 'my_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            ),
            keras.callbacks.CSVLogger(os.path.join(self.results_base_dir, 'history.csv')),
            keras.callbacks.ReduceLROnPlateau(
                patience=10,
                monitor='val_loss',
                factor=0.75,
                verbose=1,
                min_lr=5e-6)
        ]
        return callbacks_list