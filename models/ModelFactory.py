from abc import ABC, abstractmethod
# from ModelOneHotAmminoacid import ModelOneHotAmminoacid
from models.ModelEmbeddingUnidirect import ModelEmbeddingUnidirect
from models.ModelEmbeddingBidirect import ModelEmbeddingBidirect
from models.ModelEmbeddingBidirectProtein import ModelEmbeddingBidirectProtein
from models.ModelOneHotProtein import ModelOneHotProtein
from models.experimental_simple_models.experiments_with_tf_keras_nn import get_compiled_model_v1
# from models.ModelOneHotUnidirect import ModelOneHotUnidirect

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
         return ModelFactory.getExperimentalModels()
   
      raise ValueError(f'ERROR: {model_name} is not allowed!')
   
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
   def getExperimentalModels():
       return get_compiled_model_v1()

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
    
    