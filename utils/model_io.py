# -*- coding: utf-8 -*-

# save and load model
# Model Persistence
import pickle
# from sklearn.externals.joblib import dump, load
try:
    from joblib import dump, load
except:
    from sklearn.externals.joblib import dump, load
    pass

# local operations
# used in simulations


def save_model(clf, path_name):
    dump(clf, path_name)



def load_model(path_name):
    return load(path_name)


# transfer operations is needed in the future !
# http://www.runoob.com/python/python-socket.html
