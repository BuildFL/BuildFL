import time 
import random 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

# original mortar data io file 
test_Str = '2018-08-29 13:00:00+00:00,60.805,39.435,213.72,0.2,70'

BUILDING_NAMES_MORTAR = [
    'arc',
    # 'brig',
    # 'miwf',
    'smoa',
    'vm3a',
    'vmif' # too much corrupted data
]

def load_dataset_mortar():
    X_train_dict = {}
    y_train_dict = {}
    X_test_dict  = {}
    y_test_dict  = {}

    X_test_list = []
    y_test_list = []
    for b in BUILDING_NAMES_MORTAR:
        X, y = form_dataset(b)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        X_train_dict[b] = X_train 
        y_train_dict[b] = y_train 
        X_test_dict [b] = X_test
        y_test_dict [b] = y_test

        X_test_list.append(X_test)
        y_test_list.append(y_test)
        pass
    X_test_global = np.concatenate(X_test_list)
    y_test_global = np.concatenate(y_test_list)
    return X_train_dict, y_train_dict, X_test_dict, y_test_dict, X_test_global, y_test_global


def form_dataset(building_name): # single building name 
    X_list = []
    y_list = []

    lines = None 
    with open('data/%s.csv' %( building_name), 'r') as f:
        lines = f.readlines()
        pass
    for s in lines:
        x, cop = get_data_point(s)
        if cop > 0:
            X_list.append(x)
            y_list.append(cop)
        pass
    # Some building's COP get very small 
    # Caused by different units of the sensor
    if building_name == 'brig':
        y_list = [ i * 10 for i in y_list]
    if building_name == 'miwf':
        y_list = [ i * 100 for i in y_list]        
    return X_list, y_list
    # pass

def get_data_point(input_Str):
    # valid = True
    data_list = input_Str.split(',')   
    try:
        age = get_day_of_year ( data_list[0] )
        return_temp = float(data_list[1])
        supply_temp = float(data_list[2] )
        water_flow  = float(data_list[3])
        power       = float(data_list[4])
        room_temp   = float(data_list[5])
        cooling_load =   (temperature_F_to_C(return_temp)  - temperature_F_to_C(supply_temp))  * water_flow * 0.42
        cop = cooling_load / (power * 100 )
        each_x = [ age, return_temp, supply_temp, water_flow, power, room_temp]
    except:
        return [], -100 
    # data cleaning 
    if power < 1:
        cop = -100 
        each_x = []
    if cooling_load < 0:
        cop = -100 
        each_x = []
    # print(cooling_load)
    # print(cop)
    return  each_x, cop # for outlier , COP = -100 

def get_day_of_year(input_Str):
    struct_time = time.strptime(input_Str[:10], "%Y-%m-%d")
    # 解析
    year  = struct_time[0]
    month = struct_time[1]
    day   = struct_time[1]
    def count(year,month,day):
        count = 0
        #判断该年是平年还是闰年
        if year%400==0 or (year%4==0 and year%100!=0):
            # print('%d年是闰年，2月份有29天！'%year)
            li1 = [31,29,31,30,31,30,31,31,30,31,30,31]
            for i in range(month-1):
                count += li1[i]
            return count+day
        else:
            # print('%d年是平年，2月份有28天！' % year)
            li2 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            for i in range(month-1):
                count += li2[i]
            return count+day
    res_day = count(year, month, day)
    return res_day

def temperature_F_to_C(fahrenheit):
    celsius = (fahrenheit - 32.0) / 1.80
    return celsius

# original data io file 

# get batched data

def get_batched_data(X, y, ratio):
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

def aggregate_dataset(dataset_list_X, label_list_y):
    X = np.concatenate(dataset_list_X)
    y = np.concatenate(label_list_y)
    return X, y 
    pass

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


# for BuildFL public version
# we only privide a mortar dataset for test 
# user can add dataset of their own 
def prepare_dataset(data_source, building_name, **parameters ):
    '''
    assert data_source in [
            'PolyU', 'HK_Island', 'Admiralty', 'SZ_Citizen_Center', 'mortar']
    if data_source == 'Admiralty':
        X_train, X_test, y_train, y_test  = prepare_dataset_Admiralty(chiller_name_list)
    elif data_source == 'HK_Island':
        X_train, X_test, y_train, y_test  = prepare_dataset_HK_Island(chiller_name_list, **parameters)
    elif data_source == 'PolyU':
        pass
    ''' 
    if data_source == 'mortar' or data_source == "Mortar":
        # prepare dataset of mortar 
        X, y = form_dataset(building_name) # building name is a str type 
        pass
    elif data_source == 'other data set':

        pass
    else: 
        # exit 
        import sys 
        sys.exit()
        pass
    return X , y 
    pass

