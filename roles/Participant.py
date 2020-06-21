# -*- coding: utf-8 -*-
import numpy as np 
from copy import deepcopy
# from utils.data_io import prepare_dataset_Admiralty, prepare_dataset_HK_Island
# from utils.learning import incremental_clustering
# from utils.learning import clustered_learning_train
# from utils.learning import clustered_learning_predict
from utils.learning import train_model
from utils.learning import NN_incremental_train # for NN 
from utils.learning import train_model_RF
from utils.learning import calc_loss_RMSE, calc_loss_MAPE

from utils.building_data_IO import prepare_dataset
from utils.building_data_IO import get_batched_data

from models.ensemble_learning import BoostingModel

class participant(object):

    def __init__(self, ID,  data_source = 'mortar', building_name = 'DEFALUT_BUILDING_NAME' ,\
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
        # prepare training data 
        X_train, y_train = get_batched_data(self.X, self.y, self.mini_batch_size)
        if self.model in ['RF' ,'rf', "Random Forest", 'random forest']:
            self.weak_estimator = train_model_RF(X_train, y_train, **parameters)
            pass 
        elif self.model in ['Ada', 'ada', 'adaboost', 'Adaboost']:
            self.__train_Adaboost(X_train, y_train)
            pass
        elif self.model in ['BT','bt', 'Boosting Tree', 'boosting tree']:
            self.__train_BoostingTree(X_train, y_train)
            pass
        elif self.model in ['NN', 'nn', 'Neural network', 'neural network']: # NN directly update global model 
            self.__train_NN(X_train, y_train)
            pass
        # end training 
        # fill in the weak estimator 
        pass
    
    
    def __train_NN(self,X, y,  **parameters):
        if self.global_model is None :
            self.global_model = train_model(X, y , 'NN', **parameters)
        else: 
            self.global_model = NN_incremental_train(X, y , self.global_model, **parameters)
            pass
        pass

    def __train_BoostingTree(self, X, y, **parameters):
        y_pred = self.global_model.predict(X)
        y_resi = BoostingModel.get_residual_error(y, y_pred)
        self.weak_estimator = train_model(X, y_resi, 'DT', ** parameters)
        pass

    def __train_Adaboost(self, X, y,  **parameters):
        y_pred = self.global_model.predict(X)
        y_resi = BoostingModel.get_residual_error(y, y_pred)
        w  = self.__cal_W( y, y_pred )
        parameters['sample_weight'] = w 
        self.weak_estimator = train_model(X, y_resi, 'DT', **parameters)
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
        
    
    def __cal_W(self,y,pred, alpha = 1.0 ):
        # from  https://github.com/px528/AdaboostExample/blob/master/Adaboost.py
        length = len(y)
        W = np.ones(length) / length
        ret=0
        new_W=[]
        for i in range(len(y)):
            new_W.append(W[i]*np.exp(-alpha*y[i]*pred[i]))
        return np.array(new_W/sum(new_W)).reshape([len(y),1])
    

    


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