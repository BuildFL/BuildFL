# -*- coding: utf-8 -*-
import numpy as np 
from copy import deepcopy
# from utils.data_io import prepare_dataset_Admiralty, prepare_dataset_HK_Island
# from utils.learning import incremental_clustering
# from utils.learning import clustered_learning_train
# from utils.learning import clustered_learning_predict
from utils.learning import calc_loss_RMSE, calc_loss_MAPE
from utils.building_data_IO import prepare_dataset
from models.ensemble_learning import BoostingModel

class participant(object):

    def __init__(self, ID,  data_source = 'mortar', building_name = 'vm3a' ,\
                model = 'BT', mini_batch_size = 0.4 ):
        # self.aggrate_method = aggrate_method
        # assert aggrate_method.lower() in ['ensemble', 'incremental', 'distributed'] # 这里之前写了个bug 
        # self.n_clusters = n_clusters
        # self.learning_model = learning_model
        self.id = ID # ID is a number 
        self.data_source = data_source
        # assert self.data_source in [
        #     'PolyU', 'HK_Island', 'Admiralty', 'Deploy']
        self.building_name = building_name
        # if self.data_source == 'Admiralty':
        #     X_train, X_test, y_train, y_test  = prepare_dataset_Admiralty(self.chiller_id_list)
        # elif self.data_source == 'HK_Island':
        #     # elimiate the repeat samples
        #     X_train, X_test, y_train, y_test  = prepare_dataset_HK_Island(self.chiller_id_list, remove_COP_10 = True, remove_Duplicates = True)
        #     pass 
        # prepare data set 
        X, y = prepare_dataset( data_source, building_name)
        # store data
        # self.X_train = X_train
        # self.X_test  = X_test
        # self.y_train = y_train
        # self.y_test  = y_test
        self.X = np.array( X )
        self.y = np.array( y )

        # get federated  model
        # self.has_global_models = False
        self.model = model 
        
        # self evaluation result
        # self.baseline_evaluated = False
        # self.baseline_RMSE = rmse_loss
        # self.baseline_MAPE = mape_loss
        # self.baseline_evaluation()

        self.global_model = None 
        self.weak_estimator = None 

        # mini batch size 
        self.mini_batch_size = mini_batch_size
        assert self.mini_batch_size > 0 and self.mini_batch_size < 1.0 
        pass
    
    # the participant train the model 
    # based on the existing global model 
    # when ensemble, use train a weak-estimator directly 
    # def local_train( self, model, **parameters)
    def train(self, **parameters): # currently we don't use train(model, parameter), because model is stored in self.model 
        if self.model in ['RF' ,'rf', "Random Forest", 'random forest']:
            
            pass
        elif self.model in ['Ada', 'ada', 'adaboost', 'Adaboost']:
            
            pass
        elif self.model in ['BT','bt', 'Boosting Tree', 'boosting tree']:
            
            pass
        elif self.model in ['NN', 'nn', 'Neural network', 'neural network']: # NN directly update global model 
            
            pass
        #  end training 
        # fill in the weak estimator 

        pass
    


    
    def set_global_model(self, input_model):
        self.global_model = input_model
        pass

    def update(self):
        if self.model in ['NN', 'nn', 'Neural network', 'neural network']:
            return deepcopy( self.global_model )
        else: 
            return deepcopy( self.weak_estimator )
        pass
        

    

    


    pass

'''

    def local_train(self, **parameters):
        # 在有 self.kmeans 这个聚类模型的背景下做 training 
        assert self.kmeans is not None # 这个assert 就是保证有聚类模型
        clfs = clustered_learning_train(self.X_train ,self.y_train, self.kmeans, self.learning_model, **parameters)
        self.models = clfs
        pass

    def baseline_evaluation(self):# return rmse and mape score  
        # if evaluated 
        if self.baseline_evaluated == True:
            return self.baseline_RMSE, self.baseline_MAPE
            pass
        kmeans , clfs = clustered_learning(self.X_train, self.y_train, cnt_cluster= self.n_clusters, model = self.learning_model, max_depth = 5)
        self.baseline_kmeans = kmeans
        self.baseline_models = clfs
        # then predict 
        y_pred = []
        for i in range(len(self.X_test)): 
            # y_predict = clustered_learning_predict(self.X_test[i], kmeans, clfs)
            y_predict = self.baseline_predict( self.X_test[i]) # only for SINGLE sample
            y_pred.append(y_predict)
            pass
        rmse_loss = calc_loss_RMSE(self.y_test, y_pred)
        mape_loss = calc_loss_MAPE(self.y_test, y_pred)
        # store result 
        self.baseline_RMSE = rmse_loss
        self.baseline_MAPE = mape_loss
        self.baseline_evaluated = True
        # return the result 
        return rmse_loss, mape_loss
        pass
''' 