import tensorflow as tf
import copy
import os
import sys
import pickle

import numpy as np

class WrapperRawModel(object):

    def __init__(self, model, params:dict, callbacks: list):
        self.model = model
        self.params = copy.deepcopy(params)
        self.callbacks = copy.deepcopy(callbacks)
        pass

    def evaluate(self, x_test, y_test) -> dict:
        scores = self.model.evaluate(x_test, y_test, verbose=0)
        results_dict = dict(zip(self.model.metrics_names, scores))
        return results_dict

    def build(self, logger):
        if logger is not None:
            summary_list: list = list()
            # self.model.summary(print_fn=lambda x: logger.info(x))
            self.model.summary(print_fn=lambda x: summary_list.append(str(x)) )
            logger.info('\n' + '\n'.join([str(xi) for xi in summary_list]))
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
        
        assert self.model != None

        print('x_train shape', X_tr.shape)
        print('y_train shape', y_tr.shape)

        callbacks_list = self.callbacks
        batch_size = self.params['batch_size']
        
        history = self.model.fit(x=X_tr, y=y_tr, epochs=epochs, shuffle=True, batch_size=batch_size,
                    callbacks=callbacks_list, validation_data=validation_data)
        trained_epochs = callbacks_list[0].stopped_epoch - callbacks_list[0].patience +1 if callbacks_list[0].stopped_epoch != 0 else epochs
        return history, trained_epochs

    def fit_generator2(self, generator, steps_per_epoch, epochs, validation_data=None, shuffle=True, callbacks_list=None):
        # Remove early stopping
        callbacks_copy: list = copy.deepcopy(self.callbacks)
        callbacks_copy = callbacks_copy[1:]

        history = self.model.fit_generator(
            generator,
            steps_per_epoch,
            epochs,
            shuffle=False,
            callbacks=callbacks_copy, # Since we have removed Early-Stopping
            validation_data=validation_data)
        return history
    
    def train(self, x_train, y_train, epochs: int = 10, batch_size: int = 32, shuffle: bool = True, validation_data: tuple = None):
        
        assert self.model != None

        print('x_train shape', x_train.shape)
        print('y_train shape', y_train.shape)

        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            validation_data=validation_data,
            shuffle=shuffle,
            # callbacks=self.callbacks
        )
        return history

    def save_weights(self):
        results_base_dir: str = self.params['result_base_dir']
        with open(os.path.join(results_base_dir, "network_params.pickle"), 'wb') as params_pickle:
            pickle.dump(self.params, params_pickle)

        self.model.save_weights(os.path.join(f"{results_base_dir}", 'my_model_weights.h5'))
        model_json = self.model.to_json()
        with open(os.path.join(f"{results_base_dir}", "model.json"), "w") as json_file:
            json_file.write(model_json)

    def predict(self,  x_test, batch_size: int = 32, verbose: int = 0) -> np.array:
        # return np.asarray([])
        return self.model.predict(
            x_test,
            batch_size=batch_size,
            verbose=verbose,
            ).ravel()

    def predict_classes(self,  x_test, batch_size: int = 32, verbose: int = 1) -> np.array:
        # return np.asarray([])
        try:
            type_api: str = self.params['meta_info']['api']
            if type_api == 'functional':
                return self._predict_classes_funcitonal_api(
                    x=x_test,
                    # batch_size=batch_size,
                    # verbose=verbose
                )
            elif type_api == 'sequential':
                return self.model.predict_classes(
                    x_test,
                )
            else:
                raise ValueError(f"ERROR: type api {type_api} not allowed.")
        except Exception as err:
            print(f"EXCEPTION-RAISED: {err}")
            sys.exit(-1)
        pass

    def _predict_classes_funcitonal_api(self, x, batch_size=32, verbose=1):
        '''Generate class predictions for the input samples
        batch by batch.
        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.
        # Returns
            A numpy array of class predictions.
        '''

        proba = self.predict(x, batch_size=batch_size, verbose=verbose)
        return np.asarray(list(map(lambda xi: 1 if xi >= 0.50 else 0, proba)), dtype=np.int)
        """
        if proba.shape[-1] > 1:
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')
        """
        pass

    def save_model(self):
        tf.keras.models.save_model(
            model=self.model,
            filepath=self.params["model_path"],
        )
        pass  

    def plot_model(self,) -> None:
        full_model_image_path: str = os.path.join('', 'model_graph.png')
        tf.keras.utils.plot_model(self.model, full_model_image_path, show_shapes=True)
    pass