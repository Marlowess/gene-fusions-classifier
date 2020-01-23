from abc import ABC, abstractmethod
# from ModelOneHotAmminoacid import ModelOneHotAmminoacid
from models.ModelEmbeddingUnidirect import ModelEmbeddingUnidirect
from models.ModelEmbeddingBidirect import ModelEmbeddingBidirect
from models.ModelEmbeddingBidirectProtein import ModelEmbeddingBidirectProtein
from models.ModelOneHotProtein import ModelOneHotProtein
from models.experimental_simple_models.experiments_with_tf_keras_nn import get_compiled_model
# from models.ModelOneHotUnidirect import ModelOneHotUnidirect
from models.experimental_simple_models import raw_models_sequentials
from models.WrapperRawModel import WrapperRawModel


class ModelFactory():
   
   @staticmethod
   def getModelByName(model_name: str, params: dict) -> object:
      
      if model_name == 'ModelEmbeddingUnidirect':
         return ModelFactory.getModelEmbeddingUnidirect(params)

      if model_name == 'ModelEmbeddingBidirect':
         return ModelFactory.getModelEmbeddingBidirect(params)
      
      if model_name == 'ModelEmbeddingBidirectProtein':
         return ModelFactory.getModelEmbeddingBidirectProtein(params)
      
      if model_name == 'ModelOneHotProtein':
         return ModelFactory.getModelOneHotProtein(params)

      if model_name == 'ModelOneHotUnidirect':
         return ModelFactory.getModelOneHotUnidirect(params)
      
      if model_name == 'ExperimentalModels':
         return ModelFactory.getExperimentalModels(params)
      
      raise ValueError(f'ERROR: {model_name} is not allowed!')
      pass

   @staticmethod
   def getModelEmbeddingBidirect(params: dict):
       return ModelEmbeddingBidirect(params)

   @staticmethod
   def getModelEmbeddingUnidirect(params: dict):
       return ModelEmbeddingUnidirect(params)
   
   @staticmethod
   def getModelEmbeddingBidirectProtein(params: dict):
      return ModelEmbeddingBidirectProtein(params)
   
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

   @staticmethod
   def getRawModelByName(params: dict, program_params: dict):

      if params['name'] == 'raw_models_sequentials':
         model, callbacks = raw_models_sequentials.get_compiled_model(params, program_params)  
      
      return WrapperRawModel(model, params, callbacks)
    
    