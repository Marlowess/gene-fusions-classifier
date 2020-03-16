from abc import ABC, abstractmethod
from models.ModelEmbeddingBidirect import ModelEmbeddingBidirect
from models.ModelOneHotProtein import ModelOneHotProtein
from models.ModelConvBidirect import ModelConvBidirect
from models.ModelBidirectDNA import ModelBidirectDNA
from models.ModelConvUnidirect import ModelConvUnidirect
from models.ModelUnidirect import ModelUnidirect
from models.ModelKMers import ModelKMers

import os
import pickle

class ModelFactory():

   """
   Static class used for providing Custom NN models.
   """
   
   @staticmethod
   def getModelByName(model_name: str, params: dict) -> object:

      """
      It is used to get a NN model specifing its name and providing
      a dictionary object with the specification of the NN model.

      Params:
         :model_name: string object, representing the name of the model
         it's required for performing train or test phases.\n
         :params: dictionary object, representing information necessay to build a selected NN model.
      Returns:
      --------
         :object: representing the custom created NN model.
      """
      
      if model_name == 'ModelEmbeddingBidirect':
         return ModelFactory.getModelEmbeddingBidirect(params)
      
      if model_name == 'ModelOneHotProtein':
         return ModelFactory.getModelOneHotProtein(params)

      if model_name == 'ModelConvBidirect':
         return ModelFactory.getModelConvBidirect(params)

      if model_name == 'ModelBidirectDNA':
         return ModelFactory.getModelBidirectDNA(params)
      
      if model_name == 'ModelConvUnidirect':
         return ModelFactory.getModelConvUnidirect(params)

      if model_name == 'ModelUnidirect':
         return ModelFactory.getModelUnidirect(params)
      
      if model_name == 'ModelKMers':
         return ModelFactory.getModelKMers(params)
      
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
   def getModelOneHotProtein(params: dict):
       return ModelOneHotProtein(params)
   
   @staticmethod
   def getModelConvUnidirect(params: dict):
       return ModelConvUnidirect(params)

   @staticmethod
   def getModelUnidirect(params: dict):
       return ModelUnidirect(params)
   
   @staticmethod
   def getOneHotEncodedLstm(params: dict):
       return ModelOneHotProtein(params)
   
   @staticmethod
   def getModelOneHotUnidirect(params: dict):
       return None # ModelOneHotUnidirect(params)
   
   @staticmethod
   def getModelKMers(params: dict):
       return ModelKMers(params) # ModelOneHotUnidirect(params)