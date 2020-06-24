# -*- coding: utf-8 -*-
import numpy as np 
from copy import deepcopy
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from utils.ensemble_model import EnsembleModel 
from utils.model_io import save_model
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

def calc_R_2_score(y_true, y_pred):
    return r2_score(y_true, y_pred)

def initialize_kmeans_model(n_cluster = 3, save_path = 'kmeans.joblib'):
    kmeans = MiniBatchKMeans(n_clusters = n_cluster)
    save_model(kmeans, path_name = save_path)
    pass

def incremental_clustering(X_train, kmeans):
    # kmeans = MiniBatchKMeans(n_clusters= cnt_cluster)
    kmeans.partial_fit(X_train)
    return kmeans
    pass

def clustered_learning_predict(input_X, kmeans_model, clfs):
    # 1. which cluster 
    c = kmeans_model.predict([input_X])[0]
    # 2. get the COP result
    res = clfs[c].predict([input_X])[0]
    return res 
    pass

def kmeans_train(X_train, n_cluster = 5):
    kmeans = KMeans(n_clusters= n_cluster)
    kmeans.fit(X_train)
    return kmeans
    pass

def clustered_learning_train(X_train, y_train, kmeans, model = 'DT', **parameters):
    # return 
    # the prediction models 
    # length = len(X_train) # why use length ??
    clfs = {}
    cnt_cluster = kmeans.n_clusters
    # the performance seems good 
    C_Xs = kmeans.predict(X_train)
    X_dict = {}
    y_dict = {}
    # split data into different cluster 
    for c in range(cnt_cluster):
        X_dict[c] = []
        y_dict[c] = []
        pass
    for i in range(len(X_train)):
        c = C_Xs[i]
        X_dict[c].append(X_train[i])
        y_dict[c].append(y_train[i])
        pass
    for c in range(cnt_cluster):
        #clf = DecisionTreeRegressor() # can use parameter to modify
        # clf.fit(X_dict[c], y_dict[c])
        clf = train_model( X_dict[c], y_dict[c], model, **parameters)
        clfs[c] = clf 
        pass
    return clfs # which is a dict 
    pass # end of function 

def calc_loss_RMSE(reals, predictions):
    # calculate loss 
    predictions = np.array(predictions).flatten()
    reals       = np.array(reals      ).flatten()
    assert predictions.shape == reals.shape
    return np.sqrt(((predictions - reals) ** 2).mean())
    pass

def calc_loss_MAPE(reals, predictions):
    predictions = np.array(predictions).flatten()
    reals       = np.array(reals      ).flatten()
    assert predictions.shape == reals.shape
    diff = np.abs(np.array(reals) - np.array(predictions))
    ratio_list = []
    length = len(diff)
    for i in range(length):
        each_diff = diff[i]
        each_real = reals[i]
        if each_real == 0: # notice : devided by zero 
            continue
        each_ratio = each_diff / each_real
        ratio_list.append(each_ratio)
        pass
    return np.mean(ratio_list)
    pass

# for centralized mode 
def train_model( X_train, y_train, model, type = 'regression' , **parameters): # currently we only support regression model ... 
    # print(parameters)
    assert type(model) == type('neural network')
    # print(model)
    model = model.lower()
    assert model in [
        'dt', 'nn', 'mlp',
        'decision tree', 'neural network', 'multi-layer perception',
        'svr',
        'adaboost', 'ada',
        'rf', 'random forest',
    ]
    if model in ['dt', 'decision tree']:
        return train_model_DT(X_train, y_train, **parameters)
    elif model in ['nn', 'neural network']:
        return train_model_NN(X_train, y_train,  **parameters) # current is implemented in pytorch 
    elif model == 'svr':
        return train_model_SVR(X_train, y_train, **parameters)
        pass
    elif model in ['ada', 'adaboost']:
        return train_model_Adaboost(X_train, y_train, **parameters)
        pass
    elif model in ['rf', 'random forest']:
        return train_model_RF(X_train, y_train, **parameters)
        pass 
    
    pass

# detailed machine learning model training func  

def train_model_RF(X_train, y_train, **parameters):
    if 'n_estimators' in parameters.keys():
        n_estimators = parameters['n_estimators']
    else: 
        n_estimators = 10
        pass
    max_depth = None 
    if 'max_depth' in parameters.keys():
        max_depth = parameters['max_depth']
        pass
    clf = RandomForestRegressor(n_estimators= n_estimators, max_depth= max_depth)
    clf.fit(X_train, y_train)
    return clf     
    pass

def train_model_Adaboost(X_train, y_train, **parameters):
    from sklearn.ensemble import AdaBoostRegressor
    if 'n_estimators' in parameters.keys():
        n_estimators = parameters['n_estimators']
    else: 
        n_estimators = 10
        pass
    clf = AdaBoostRegressor(n_estimators= n_estimators)
    clf.fit(X_train, y_train)
    return clf 
    pass

def train_model_DT(X_train, y_train, **parameters):
    max_depth = 10
    if 'max_depth' in parameters.keys():
        max_depth = parameters['max_depth']
        pass
    sample_w = None
    if 'sample_weight' in parameters.keys():
        sample_w = parameters['sample_weight']
        pass
    clf = DecisionTreeRegressor(max_depth= max_depth) 
    # for now we use the default parameter 
    # it acts good 
    clf.fit(X_train, y_train, sample_weight= sample_w)
    return clf
    # pass

def train_model_NN(X_train, y_train, **paramater_dict):
    y_train = deepcopy(y_train)
    X_train = deepcopy(X_train)
    # print(paramater_dict)
    # default parameter 
    # print('NN')
    import torch 
    use_cuda = torch.cuda.is_available()
    lr = 0.001
    epoch = 2000 
    if 'lr' in paramater_dict.keys():
        lr = paramater_dict['lr']
        pass
    if 'epoch' in paramater_dict.keys():
        epoch = paramater_dict['epoch']
        pass
    # print(lr)
    # now use pytorch model, has [predict] function 
    # 1. get feature number 
    feature_number = np.array(X_train).shape[-1] 
    # print('feature number', feature_number)
    # 2. initial model
    from models.NN import RegressionNN
    from torch import nn
    model = RegressionNN(feature_number)
    # 3. convert to tensor
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_train = y_train.reshape( (len(y_train),1) )
    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    if use_cuda:
        X_train, y_train = X_train.cuda(), y_train.cuda()
        model = model.cuda()
    # 4. define loss 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr )
    # 5. train 
    for i in range(epoch):
        y_pred = model.forward(X_train) # 算输出
        loss = criterion(y_pred,y_train)   # 算loss 
        optimizer.zero_grad()        # 清除梯度记录
        loss.backward()                # 反向传播
        optimizer.step()                # 更新参数
        pass
    # print('Done')
    # 6. return 
    if use_cuda:
        model = model.cpu()
        # 之后test 没必要GPU
    return model
    pass

def train_model_SVR(X_train, y_train):
    from sklearn.svm import SVR 
    model = SVR(gamma = 'scale')
    model.fit(X_train, y_train)
    return  model 
    pass

def NN_incremental_train( X_train, y_train,nn_model , **paramater_dict):
    import torch 
    y_train = deepcopy(y_train)
    X_train = deepcopy(X_train)
    use_cuda = torch.cuda.is_available()
    lr = 0.001
    epoch = 2000 
    if 'lr' in paramater_dict.keys():
        lr = paramater_dict['lr']
        pass
    if 'epoch' in paramater_dict.keys():
        epoch = paramater_dict['epoch']
        pass
    # print(lr)
    # now use pytorch model, has [predict] function 
    # 1. get feature number 
    # feature_number = np.array(X_train).shape[-1] 
    # print('feature number', feature_number)
    # 2. get model
    # from models.NN import RegressionNN
    from torch import nn
    # model = RegressionNN(feature_number)
    model = nn_model
    # 3. convert to tensor
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    # print( y_train.shape )
    # print( y_train )
    y_train = y_train.reshape( (len(y_train), 1) )  
    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    if use_cuda:
        X_train, y_train = X_train.cuda(), y_train.cuda()
        model = model.cuda()
    # 4. define loss 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr )
    # 5. train 
    for i in range(epoch):
        y_pred = model.forward(X_train) # 算输出
        loss = criterion(y_pred,y_train)   # 算loss 
        optimizer.zero_grad()        # 清除梯度记录
        loss.backward()                # 反向传播
        optimizer.step()                # 更新参数
        pass
    # print('Done')
    # 6. return 
    if use_cuda:
        model = model.cpu()
        # 之后test 没必要GPU
    return model    


    pass 

