from abc import ABC, abstractmethod
# from ModelOneHotAmminoacid import ModelOneHotAmminoacid
from ModelEmbeddedLstmOneLayer import ModelEmbeddedLstmOneLayer

class ModelFactory():
    def getEmbeddingBiLstmAttentionProtein(params: dict):
       pass
    def getEmbeddingBiLstmAttentionDna(params: dict):
       pass
    def getOneHotEncodedLstm(params: dict):
       # return ModelOneHotProtein(params)
       pass
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
    
    