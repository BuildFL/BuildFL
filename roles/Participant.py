# -*- coding: utf-8 -*-
import numpy as np 
from copy import deepcopy
from utils.data_io import prepare_dataset_Admiralty, prepare_dataset_HK_Island
from utils.learning import incremental_clustering
from utils.learning import clustered_learning_train
from utils.learning import clustered_learning_predict
from utils.learning import calc_loss_RMSE, calc_loss_MAPE

# build base line model for each participant 
from speed_up_clustered_learning import clustered_learning

# 在代码层次的下一步抽象：
# 1. 通过参数决定model的类型（dt，还是NN，还是adaboost）
# 2. 通过参数决定聚合model 的方法，并且在这里检查是否合理，比如dt就没法接力训练
# 3. 把上述的函数在类的外部实现，在类内调用, 位于 utils.learning 中
# 4. server 类也需要实现出来
# 5. 需要解决cluster 聚不到数据时候的训练问题，没数据肯定不训练了，定义一个 《空模型》类，便于识别吧

class participant(object):

    def __init__(self, ID, chiller_id_list, data_source = 'PolyU', \
                learning_model = 'DT', aggrate_method= 'Ensemble', n_clusters = 5 ):
        self.aggrate_method = aggrate_method
        assert aggrate_method.lower() in ['ensemble', 'incremental', 'distributed'] # 这里之前写了个bug 
        self.n_clusters = n_clusters
        self.learning_model = learning_model
        self.id = ID # is a number 
        self.data_source = data_source
        assert self.data_source in [
            'PolyU', 'HK_Island', 'Admiralty', 'Deploy']
        self.chiller_id_list = chiller_id_list
        if self.data_source == 'Admiralty':
            X_train, X_test, y_train, y_test  = prepare_dataset_Admiralty(self.chiller_id_list)
        elif self.data_source == 'HK_Island':
            # elimiate the repeat samples
            X_train, X_test, y_train, y_test  = prepare_dataset_HK_Island(self.chiller_id_list, remove_COP_10 = True, remove_Duplicates = True)
            pass # to be continued 
        # store data
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

        # get federated  model
        self.has_federated_models = False
        
        # self evaluation result
        self.baseline_evaluated = False
        # self.baseline_RMSE = rmse_loss
        # self.baseline_MAPE = mape_loss
        # 还有 baseline_kmeans 和 baseline_models 
        self.baseline_evaluation()
        pass
    
    # 待完成的功能，现在看来 ensemble 方法还可以
    # 先往后推进吧，我觉得
    def incremental_train(self, **parameters ):
        # 这里适用于 NN
        assert self.learning_model.lower() in ['nn', 'neural network']
        # when aggrate method is incremental_train, this 
        # method incrementally train the model 
        # the federated model should be self.federated_models
        # which is a dict having self.n_cluster keys
        if self.has_federated_models == False:
            # no federated learning 
            # participant in cold start mode
            self.local_train(**parameters)
            self.federated_models = self.models
            self.has_federated_models = True
            pass
        # 本函数现在暂缓开发，毕竟涉及到不少细节功能，现在也无心一一实现
        # ensemble 如果表现可以，就先这么用着，毕竟还有 TFL 和 sequence 要弄
        else: # then  
            # 1. extract parameters , delete model 

            # 2. initial new model <<-- parameters 

            # 3 training
 
            # 4. return model 
            pass


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

    def set_kmeans_model(self, kmeans): 
        self.kmeans = deepcopy( kmeans ) # use deep copy to get a unique object (ID)
        pass
    
    def set_federated_models(self, input_model_dict):
        self.federated_models = input_model_dict 
        self.has_federated_models = True
        pass
    
    # update self.kmeans 
    def local_clustering(self):
        self.kmeans = incremental_clustering(self.X_train, self.kmeans)
        pass
    
    def local_train(self, **parameters):
        # 第一次训练
        # 建立新的模型
        # 在有 self.kmeans 这个聚类模型的背景下做 training 
        assert self.kmeans is not None # 这个assert 就是保证有聚类模型
        clfs = clustered_learning_train(self.X_train ,self.y_train, self.kmeans, self.learning_model, **parameters)
        self.models = clfs
        pass
    
    def baseline_predict(self, input_X): # single sample 
        res = clustered_learning_predict(input_X, self.baseline_kmeans, self.baseline_models)
        return res 
        pass

    def local_learning_predict(self, input_X): 
        # use federated cluster model and local learning model 
        res = clustered_learning_predict(input_X, self.kmeans, self.models)
        return res 
        pass
    
    def federated_predict(self, input_X): # single sample 
        # use all federated cluster and learning model
        assert self.has_federated_models == True  # exists
        res = clustered_learning_predict(input_X, self.kmeans, self.federated_models)
        return res 
        pass
    
    def federated_evaluation(self): # predict participant's own dataset
        # but use all federated model  (learning and clustering )
        y_pred = []
        for i in range(len(self.X_test)): 
            # y_predict = clustered_learning_predict(self.X_test[i], kmeans, clfs)
            y_predict = self.federated_predict( self.X_test[i]) # only for SINGLE sample
            y_pred.append(y_predict)
            pass
        rmse_loss = calc_loss_RMSE(self.y_test, y_pred)
        mape_loss = calc_loss_MAPE(self.y_test, y_pred)
        return rmse_loss, mape_loss
        pass
    
    def compare(self):
        print('Participant ID: %d' %(self.id))
        print('Baseline:', self.baseline_evaluation())
        print('Federated:', self.federated_evaluation())
        print('\n')
        pass

    pass