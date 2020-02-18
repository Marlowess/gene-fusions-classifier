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
from models.ModelConvUnidirect import ModelConvUnidirect


from models import sequence_oriented_model
import os
import pickle

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
      
      if model_name == 'ModelConvUnidirect':
         return ModelFactory.getModelConvUnidirect(params)
      
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
   def getOneHotEncodedLstm(params: dict):
       return ModelOneHotProtein(params)
   
   @staticmethod
   def getModelOneHotUnidirect(params: dict):
       return None # ModelOneHotUnidirect(params)

   @staticmethod
   def getExperimentalModels(params: dict):
       return get_compiled_model(params)

   @staticmethod
   def _check_for_prentrained_model(params):
      if 'result_base_dir' in params.keys():
         results_base_dir = params['result_base_dir']
      else:
         results_base_dir = None

      if 'only-test' in params.keys():
         only_test = params['only-test']
      else:
         only_test = False

      pretrained_model = params.get('pretrained_model', None)    
      if pretrained_model is not None:
            print("loading model")
            train_dir = "/"
            train_dir = train_dir.join(params['pretrained_model'].split("/")[:-1])                                              
            print(train_dir)
            with open(os.path.join(train_dir, "network_params.pickle"), 'rb') as params_pickle:
               params = pickle.load(params_pickle)
            params['result_base_dir'] = results_base_dir
      params['only-test'] = only_test
      return params

   @staticmethod
   def getRawModelByName(params: dict, program_params: dict):

      model_name = params['name']
      params = ModelFactory._check_for_prentrained_model(params)

      if model_name == 'raw_models_sequentials':
         model, callbacks = raw_models_sequentials.get_compiled_model(params, program_params)

      elif model_name == 'model_dna_embedding_unidirect':
         model, callbacks = model_dna_embedding_unidirect.get_compiled_model(params, program_params)
      
      elif model_name == 'sequence_oriented_model':
         model, callbacks = sequence_oriented_model.get_compiled_model(params, program_params)
      
      else:
         raise ValueError(f"ERROR: {model_name} is not allowed!")       
      return WrapperRawModel(model, params, callbacks)
    
    
