import time 
import numpy as np 
from sklearn.model_selection import train_test_split

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
    with open('mortar_dataset/%s.csv' %( building_name), 'r') as f:
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