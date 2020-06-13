try: 
    from utils.data_io import prepare_dataset_HK_Island, prepare_dataset
except:
    import sys
    sys.path.append("../") 
    from utils.data_io import prepare_dataset_HK_Island, prepare_dataset

CHILLER_NAMES_HK_SILAND = [
        "CP1",  "CP4",  "CPN",  "CPS",  "DEH",  "DOH",  "LIH",  "OIE",  "OXH"
    ]

SIMPLIFIED_CHILLER_NAMES_HK_SILAND = [
        "CP1",  "CP4",     "CPS",  "DEH", "LIH",  "OXH" 
    ]

def generate_full_permutation(elements):
    assert type(elements) == type( [] )
    res = []
    def dfs(curr , rest ):
        if len(rest ) == 0:
            res.append(curr)
            return 
        for r in rest :
            rest_tmp = rest.copy()
            rest_tmp.remove(r)
            dfs( curr + [r] , rest_tmp )
        pass
    dfs([] , elements )
    return res 



import random 
def generate_random_data_order(elements ):
    order_dict = {}
    for each_building in elements:
        X_train, _, _, _  = prepare_dataset('HK_Island',[ each_building ], remove_Duplicates = True )
        length = len(X_train)
        order_tmp = list(range(length))
        random.shuffle( order_tmp)
        order_dict[each_building] = order_tmp
        pass
    return order_dict
    pass

import pickle 
def get_permutations():
    try:
        with open( 'results/experiments/chiller_permutations.pickle', 'rb') as file :
            res = pickle.load(file)
        return res 
    except:
        with open( '../results/experiments/chiller_permutations.pickle', 'rb') as file :
            res = pickle.load(file)
        return res 


def get_order_dict():
    try:
        with open( 'results/experiments/order_dict.pickle', 'rb') as file:
            res = pickle.load(file)
        return res
    except:
        with open( '../results/experiments/order_dict.pickle', 'rb') as file:
            res = pickle.load(file)
        return res         


## code to dump them to disk 
'''
chiller_permutations = generate_full_permutation( SIMPLIFIED_CHILLER_NAMES_HK_SILAND )
order_dict = generate_random_data_order(SIMPLIFIED_CHILLER_NAMES_HK_SILAND)

import pickle
f_order = open('order_dict.pickle', 'wb')
pickle.dump( order_dict , f_order)
f_order.close()

f_permutations = open('chiller_permutations.pickle', 'wb')
pickle.dump( chiller_permutations, f_permutations)
f_permutations.close()
''' 

# this is for test 
# print( permutation( [1,2,3] ) ) 
