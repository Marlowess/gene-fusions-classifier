from abc import ABC, abstractmethod
from models.ModelEmbeddingUnidirect import ModelEmbeddingUnidirect
from models.ModelEmbeddingBidirect import ModelEmbeddingBidirect
from models.ModelOneHotProtein import ModelOneHotProtein
from models.ModelConvBidirect import ModelConvBidirect
from models.experimental_simple_models.experiments_with_tf_keras_nn import get_compiled_model
from models.experimental_simple_models import raw_models_sequentials
from models.experimental_simple_models import model_dna_embedding_unidirect
from models.WrapperRawModel import WrapperRawModel
from models.ModelBidirectDNA import ModelBidirectDNA


from models import sequence_oriented_model

class ModelFactory():
    @staticmethod
    def getModelByName(model_name: str, params: dict) -> object:

        if model_name == 'ModelEmbeddingUnidirect':
            return ModelFactory.getModelEmbeddingUnidirect(params)

        if model_name == 'ModelEmbeddingBidirect':
            return ModelFactory.getModelEmbeddingBidirect(params)

        if model_name == 'ModelOneHotProtein':
            return ModelFactory.getModelOneHotProtein(params)

        if model_name == 'ModelOneHotUnidirect':
            return ModelFactory.getModelOneHotUnidirect(params)

        if model_name == 'ModelConvBidirect':
            return ModelFactory.getModelConvBidirect(params)

        if model_name == 'ModelBidirectDNA':
            return ModelFactory.getModelBidirectDNA(params)

        if model_name == 'ExperimentalModels':
            return ModelFactory.getExperimentalModels(params)

        raise ValueError(f'ERROR: {model_name} is not allowed!')

    @staticmethod
    def getModelBidirectDNA(params):
        return ModelBidirectDNA(params)

    @staticmethod
    def getModelConvBidirect(params):
        return ModelConvBidirect(params)

    @staticmethod
    def getModelEmbeddingBidirect(params: dict):
        return ModelEmbeddingBidirect(params)

    @staticmethod
    def getModelEmbeddingUnidirect(params: dict):
        return ModelEmbeddingUnidirect(params)

    @staticmethod
    def getModelOneHotProtein(params: dict):
        return ModelOneHotProtein(params)

    @staticmethod
    def getOneHotEncodedLstm(params: dict):
        return ModelOneHotProtein(params)

    @staticmethod
    def getModelOneHotUnidirect(params: dict):
        return None # ModelOneHotUnidirect(params)

    @staticmethod
    def getExperimentalModels(params: dict):
        return get_compiled_model(params)
