import time 
# import random
import numpy as np 

from utils.learning import train_model, NN_incremental_train

from utils.building_data_IO import BUILDING_NAMES_MORTAR
from utils.building_data_IO import prepare_dataset

from roles.Participant import participant
from roles.Server import parameter_server

from models.ensemble_learning import BoostingModel, EnsembleModel

for building_name in BUILDING_NAMES_MORTAR:
    X, y = prepare_dataset('Mortar', building_name)
    pass

p = participant(1, 'mortar', 'vm3a', model= 'rf')

ps = parameter_server(model= 'rf')


b = BoostingModel()
e = EnsembleModel()

rf_model = train_model( X, y, model= 'rf')

nn_model = train_model( X, y, model = 'NN')
nn_model = NN_incremental_train(X, y , nn_model) # 顺序有一些不一样

