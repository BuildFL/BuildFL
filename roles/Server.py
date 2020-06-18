# -*- coding: utf-8 -*-
import numpy as np 
from copy import deepcopy 

from models.ensemble_learning import BoostingModel, EnsembleModel
# from utils.building_data_IO import prepare_dataset # Server CANNOT access data set !

class parameter_server(object):
    '''
        Parameter Server (PS) class in Federated Learning 
    '''
    def __init__(self, model = 'rf'):
        # self.n_cluster = n_cluster
        # self.aggrate_method = aggrate_method.lower() # ensemble , average
        self.global_model = None 
        self.model = model # model type 
        assert model in [
            'RF' ,'rf', "Random Forest", 'random forest',
            'Ada', 'adaboost', 'Adaboost', 'ada',
            'BT', 'bt', 'Boosting Tree', 'boosting tree',
            'NN', 'nn', 'Neural network', 'neural network']
        if model in ['Ada', 'adaboost', 'Adaboost', 'ada', 'BT','bt', 'Boosting Tree', 'boosting tree']:
            self.global_model = BoostingModel()
            pass 
        if model in [ 'RF' ,'rf', "Random Forest", 'random forest',]:
            self.global_model = EnsembleModel()
            pass
        pass

    def aggregate_model(self, input_model):
        if self.model in ['RF' ,'rf', "Random Forest", 'random forest']:
            # assert type(input_model) == type( [] ) # is a list  
            self.global_model.add_model( input_model ) # use add model under RF 
            pass
        elif self.model in ['Ada', 'adaboost', 'Adaboost', 'ada']:
            self.global_model.aggregate( input_model )
            pass
        elif self.model in ['BT','bt', 'Boosting Tree', 'boosting tree']:
            self.global_model.aggregate( input_model )
            pass
        elif self.model in ['NN', 'nn', 'Neural network', 'neural network']: # for NN, PS can replace the model since the participant update its full model 
            self.global_model = deepcopy( input_model ) 
            pass

        pass
    
    def update(self): # when this happens at server, maybe we should call it `distribute` ? 
        return self.distribute()

    def distribute(self): 
        return deepcopy(self.global_model)




    pass