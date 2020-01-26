import tensorflow as tf
import copy
import os

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
            self.model.summary(print_fn=lambda x: logger.info(x))
        else:
            self.model.summary()
    
    def train(self, x_train, y_train, epochs: int = 10, batch_size: int = 32, shuffle: bool = True, validation_data: tuple = None):
        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            validation_data=validation_data,
            shuffle=shuffle,
            callbacks=self.callbacks
        )
        return history

    def plot_model(self,) -> None:
        full_model_image_path: str = os.path.join('', 'model_graph.png')
        tf.keras.utils.plot_model(self.model, full_model_image_path, show_shapes=True)
    pass