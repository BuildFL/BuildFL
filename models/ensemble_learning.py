# -*- coding: utf-8 -*-
import numpy as np 
from copy import deepcopy
# this file contains models in HVAC 

class EnsembleModel(object):
    # predict 结果平均
    def __init__(self, model_list = [] ): # model_list = []):
        assert type(model_list) == type([])
        self.model_list = model_list
        # self.estimator_weight_list = []
        pass
    
    def add_model(self, input_clf): # , estimator_weight = 1.0):
        model = deepcopy(input_clf)
        self.model_list.append(model)
        # self.estimator_weight_list.append(estimator_weight)
        pass
    
    def predict(self, input_X):
        assert len(self.model_list) > 0 
        if len(self.model_list) == 1:
            res = np.array ( self.model_list[0].predict(input_X) )
            pass
        else:
            results = []
            length_of_models = len(self.model_list)
            for i in range(length_of_models):
                each_model = self.model_list[i]
                y_pred = each_model.predict(input_X)
                results.append(y_pred )
                pass
            res = np.mean(results, axis = 0 )
        assert len(res) == len(input_X)
        return res 

    pass # end of the class 

class BoostingModel(EnsembleModel):

    @staticmethod
    def get_residual_error(real_values, predictions):
        real = np.array(real_values).flatten()
        pred = np.array(predictions).flatten()
        assert len(pred) == len(real)
        res = []
        length = len(real)
        for i in range(length):
            r = real[i] - pred[i]
            res.append(r)
            pass
        return np.array( res )
        pass

    def clear(self):
        self.model_list = []
        pass
    
    # remove the last added model 
    def undo(self):
        self.model_list = self.model_list[: -1]
        pass

    def predict(self, input_X):
        if len(self.model_list) == 0: 
            res = np.zeros( len(input_X) ) # this is for boosting 
            return res 

        if len(self.model_list) == 1:
            res = np.array ( self.model_list[0].predict(input_X) )
            return res 
            pass
        res = []
        length = len(input_X)
        
        prediction_list = []
        for each_model in self.model_list:
            res_tmp = each_model.predict(input_X)
            prediction_list.append( res_tmp )
            pass
        for i in range(length):
            y = 0
            for j in range(len(self.model_list)):
                y += prediction_list[j][i]
                pass
            res.append(y)
            pass
        assert len(res) == len(input_X) 
        return np.array( res ) 
        pass