import time 
import random
import numpy as np 

from settings import TOTAL_EPOCH_CNT, MODEL, DATA_SOURCE
from utils.building_data_IO import BUILDING_NAMES_MORTAR


from utils.learning import train_model # , NN_incremental_train
from utils.learning import calc_loss_RMSE

from utils.building_data_IO import prepare_dataset

from roles.Participant import participant
from roles.Server import parameter_server

from models.ensemble_learning import BoostingModel, EnsembleModel

if MODEL == 'rf':
    TOTAL_EPOCH_CNT = 1 

participant_list = []
ps = parameter_server(model= MODEL)

# initialize Participants 
for i in range( len(BUILDING_NAMES_MORTAR) ):
    one_participant = participant(i, DATA_SOURCE, BUILDING_NAMES_MORTAR[i], MODEL)
    participant_list.append(one_participant) 
    pass

# simulated federated learning training process 
# if RF 
for p in participant_list:
    # p = random.choice( participant_list)
    p.train()
    weak_estimator = p.update()
    ps.aggregate_model(weak_estimator)
    pass

global_model = ps.global_model

# for evaluation 
for p in participant_list:
    p.set_global_model(global_model)
    p.evaluate()
    pass
print('\n')

# iterative 
TOTAL_EPOCH_CNT = 10

participant_list = []
ps = parameter_server(model= 'nn')

# initialize Participants 
for i in range( len(BUILDING_NAMES_MORTAR) ):
    one_participant = participant(i, DATA_SOURCE, BUILDING_NAMES_MORTAR[i], MODEL)
    participant_list.append(one_participant) 
    pass

for i in range( TOTAL_EPOCH_CNT ):
    p.set_global_model(ps.global_model) 
    p = random.choice(participant_list) # select the participant 
    p.train()
    weak_estimator = p.update()
    ps.aggregate_model(weak_estimator)
    pass 

# for evaluation 
for p in participant_list:
    p.set_global_model(global_model)
    p.evaluate()
    pass
