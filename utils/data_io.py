# Author: GUO Yunzhe 

import numpy as np 
import time
import pandas as pd 
from sklearn.model_selection import train_test_split
import random 

# 地理信息
geo = {}
geo['Cityplaza One'] = (22.285799, 114.217880)
geo['Cityplaza Three'] = (22.287819, 114.216815)
geo['Cityplaza Four'] = (22.287720, 114.217932)
geo['Cityplaza Mall'] = (22.286673, 114.217413)
geo['Cityplaza Carpark'] = (22.286737, 114.218101)

geo['Devon House'] = (22.287269, 114.211015)
geo['Dorset House'] = (22.286994, 114.211786)
geo['Lincoln House'] = (22.287724, 114.212330)
geo['One Island East'] = (22.286128, 114.213332)
geo['Oxford House'] = (22.287155, 114.213632)


FEATURE_NAMES_Admiralty = [
    'TRTEMP', 'TELE', 'TFRATE','TAGE',
    'TCL', 'TOT','TAELE', 'TPELE',
]

DATASET_FEATURE_NAMES_Admiralty = FEATURE_NAMES_Admiralty

def get_part_data(X, y, index, cnt_parts):
    # index from 0 -> cnt_parts-1
    length = len(X)
    assert length == len(y)
    interval = int( (length / float(cnt_parts)) - 1  )
    st = (index ) * interval
    ed = (index + 1) * interval
    return X[st:ed], y[st:ed]
    pass

def get_sample_data(X, y, ratio):
    assert ratio <= 1.0 
    assert ratio > 0
    assert len(X) == len(y)
    if not type(X) is np.array( X ):
        X = np.array(X)
        y = np.array(y)
        pass
    index_list = list( range( len(X )) )
    random.shuffle(index_list)
    required_length = int( ratio * len(X) )
    required_index = index_list[:required_length]
    assert type(required_index) == type([]) 
    return X[required_index],y[required_index]
    pass

def get_chiller_data_Admiralty(chiller_Number):
    # chiller number from 0 - 4 
    i = chiller_Number
    X = np.load('Data_Admiralty/chiller_X_%d.npy' % i)
    y_COP = np.load('Data_Admiralty/chiller_COP_%d.npy' % i)
    # y_TPCOP = np.load('data/chiller_TPCOP_%d.npy' % i)
    return X, y_COP #, y_TPCOP
    pass


def prepare_dataset_Admiralty(chiller_No):
    # prepare dataset 
    X_list = []
    y_list = []
    for i in chiller_No:
        X, y = get_chiller_data_Admiralty(i)
        X_list.append(X)
        y_list.append(y)
        pass
    X, y = aggregate_dataset(X_list, y_list)
    X_train, X_test, y_train, y_test = train_test_split(\
        X, y, test_size=0.2) # , random_state=42) # no random state !
    return X_train, X_test, y_train, y_test
    pass


def aggregate_dataset(dataset_list_X, label_list_y):
    X = np.concatenate(dataset_list_X)
    y = np.concatenate(label_list_y)
    return X, y 
    pass

CHILLER_NAMES_HK_SILAND = [
    "CP1",  "CP4",  "CPN",  "CPS",  "DEH",  "DOH",  "LIH",  "OIE",  "OXH",
]
CHILLER_NAMES_HK_ISLAND = [
    "CP1",  "CP4",  "CPN",  "CPS",  "DEH",  "DOH",  "LIH",  "OIE",  "OXH",
]

BUILDING_NAMES_HK_ISLAND = CHILLER_NAMES_HK_ISLAND

FEATURE_NAMES_HK_ISLAND = [
    "building","chillerName",
    "time","coolingLoad",
    "flowRate","returnTemp",
    "supplyTemp","R2",
    "temperature","year",
    "month","day",
    "clock","cop","age"
]

# 为什么要改这里：
# 因为 COP = 10 是开始处理错误点的方法，但是后来感觉，这个处理方法也不是很妥当
# 可能对model的训练有负面影响
# 现在直接移除错误点试一下
def prepare_dataset_HK_Island(chiller_name_list, **parameters):
    remove_COP_10 = True
    remove_Duplicates = True
    if 'remove_COP_10'in parameters.keys():
        remove_COP_10 = parameters['remove_COP_10']
        pass
    if 'remove_Duplicates' in parameters.keys():
        remove_Duplicates = parameters['remove_Duplicates']
        '''
        if remove_Duplicates == False:
            print("NOT remove duplicates")
            pass 
        '''
        pass
    X_list = []
    y_list = []
    for i in chiller_name_list:
        X, y = get_chiller_data_HK_Island(i)
        X_list.append(X)
        y_list.append(y)
        pass
    X, y = aggregate_dataset(X_list, y_list)
    length = len(y)
    if remove_COP_10 == True:
        # print('test: remove the 10 COP data point')
        index_list_COP_10 = []
        # 1. record index where y==10
        for i in range(length):
            label = y[i]
            if label == 10:
                index_list_COP_10.append(i) 
                pass
            pass
        # now we have index list
        # 2. delete the 10 data-point 
        X = np.delete(X, index_list_COP_10, axis = 0)
        y = np.delete(y, index_list_COP_10, axis = 0)
        pass
    if remove_Duplicates:
        X, y = remove_duplicate(X, y)
        pass
    X_train, X_test, y_train, y_test = train_test_split(\
        X, y, test_size=0.2) 
    return X_train, X_test, y_train, y_test

def remove_duplicate(X, y):
    res_X = []
    res_y = []
    length = len(X)
    assert length == len(y)
    hash_table = {}
    for i in range(length):
        x_tmp = X[i]
        y_tmp = y[i]
        val = hash_table.get(y_tmp, 0)
        if val == 0: # y not exist in hash table 
            res_y.append(y_tmp)
            res_X.append(x_tmp)
            hash_table[val] = 1
        else: 
            # do nothing 
            pass
        pass
    return np.array(res_X), np.array(res_y)
    pass

def get_chiller_data_HK_Island(Chiller_Name):
    assert Chiller_Name in CHILLER_NAMES_HK_ISLAND
    file_path = 'Data_HK_Island/' + Chiller_Name + '.csv'
    # df = pd.read_csv(file_path, names= FEATURE_NAMES_HK_ISLAND, dtype= str)
    df = pd.read_csv(file_path)
    # I remember there is 6 features can be used in hk islang dataset 
    y_COP =  np.array(df['cop'] , dtype = np.double)
    # useful features 
    # "coolingLoad", "flowRate","returnTemp","supplyTemp","R2","age"
    # 6 features 
    cl = df['coolingLoad']
    fr = df['flowRate']
    rt = df['returnTemp']
    st = df['supplyTemp']
    r2 = df['R2']
    ag = df['age']
    assert len(cl) == len(fr) == len(rt) == len(st) == len(r2) == len(ag)
    X = []
    for i in range( len(cl) ):
        xx_tmp = [
            cl[i],
            fr[i],
            rt[i],
            st[i], 
            r2[i], 
            ag[i]
        ]
        X.append(xx_tmp)
        pass
    X = np.array(X, dtype = np.float64)
    assert len(X) == len(y_COP)
    return X, y_COP, 
    pass

def prepare_dataset(data_source, chiller_name_list, **parameters ):
    assert data_source in [
            'PolyU', 'HK_Island', 'Admiralty', 'Deploy']
    if data_source == 'Admiralty':
        X_train, X_test, y_train, y_test  = prepare_dataset_Admiralty(chiller_name_list)
    elif data_source == 'HK_Island':
        X_train, X_test, y_train, y_test  = prepare_dataset_HK_Island(chiller_name_list, **parameters)
    elif data_source == 'PolyU':
        pass
    else: 
        # deploy ， 可能还需要从数据库中查询之类的？
        pass
    return X_train, X_test, y_train, y_test
    pass

