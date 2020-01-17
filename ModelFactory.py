from abc import ABC, abstractmethod
# from ModelOneHotAmminoacid import ModelOneHotAmminoacid
from ModelEmbeddedLstmOneLayer import ModelEmbeddedLstmOneLayer
from ModelEmbeddingBidirectProtein import ModelEmbeddingBidirectProtein
from ModelOneHotProtein import ModelOneHotProtein

class ModelFactory():
   
   @staticmethod
   def getModelByName(model_name: str, params: dict) -> object:
      if model_name == 'EmbeddingBiLstmAttentionProtein':
         return ModelFactory.getEmbeddingBiLstmAttentionProtein(params)
      
      if model_name == 'EmbeddingBiLstmAttentionDna':
         return ModelFactory.getEmbeddingBiLstmAttentionDna(params)
      
      if model_name == 'OneHotEncodedLstm':
         return ModelFactory.getOneHotEncodedLstm(params)

      if model_name == 'EmbeddingLstm':
         return ModelFactory.getEmbeddingLstm(params)
      
      if model_name == "ModelEmbeddingBidirectProtein":
         return ModelFactory.getModelEmbeddingBidirectProtein(params)
      raise ValueError(f'ERROR: {model_name} is not allowed!')
   
   @staticmethod
   def getEmbeddingBiLstmAttentionProtein(params: dict):
       return None
   
   @staticmethod
   def getModelEmbeddingBidirectProtein(params: dict):
      return ModelEmbeddingBidirectProtein(params)
   
   @staticmethod
   def getEmbeddingBiLstmAttentionDna(params: dict):
       return None
   
   @staticmethod
   def getOneHotEncodedLstm(params: dict):
       return ModelOneHotProtein(params)
   
   @staticmethod
   def getEmbeddingLstm(params: dict):
       return ModelEmbeddedLstmOneLayer(params)

# class AbstractModel(ABC):
#     @abstractmethod
#     def __init__(self):
#         # super().__init__()
#         pass
    
#     @abstractmethod
#     def build():
#         pass
    
#     @abstractmethod
#     def fit():
#         pass
    
#     @abstractmethod
#     def evaluate():
#         pass
    
    