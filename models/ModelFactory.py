from abc import ABC, abstractmethod
from models.ModelEmbeddingBidirect import ModelEmbeddingBidirect
from models.ModelOneHotProtein import ModelOneHotProtein
from models.ModelConvBidirect import ModelConvBidirect
from models.ModelBidirectDNA import ModelBidirectDNA
from models.ModelConvUnidirect import ModelConvUnidirect
from models.ModelUnidirect import ModelUnidirect

import os
import pickle

class ModelFactory():
   
   @staticmethod
   def getModelByName(model_name: str, params: dict) -> object:
      
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