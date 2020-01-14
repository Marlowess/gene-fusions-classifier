from abc import ABC, abstractmethod
from ModelOneHotAmminoacid import ModelOneHotAmminoacid

class ModelFactory():
    def getEmbeddingBiLstmAttentionProtein(params):
       pass
    def getEmbeddingBiLstmAttentionDna(params):
       pass
    def getOneHotEncodedLstm(params):
       return ModelOneHotProtein(params)
    def getEmbeddingLstm(params):
        pass

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
    
    