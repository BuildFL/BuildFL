
import time, random 
from utils.learning import train_model, calc_loss_RMSE

from utils.ensemble_model import EnsembleModel, BoostingModel
from utils.quick_load_for_comparision import load_comparasion_dataset
from utils.data_io import get_sample_data
from evaluation_tool import basic_evaluate, write_record_to_file
# we have BoostingModel.get_residual_error

from utils.mortar_tool import load_dataset_mortar, prepare_dataset_mortar,  basic_evaluate_mortar , BUILDING_NAMES_MORTAR

from faults.feature_removal import remove_feature # 0 - 5 
REMOVE_FEATURE_INDEX = 5

dt_depth = 8

X_train_dict, y_train_dict, X_test_dict, y_test_dict, X_test_global, y_test_global = load_dataset_mortar()

# distribuetd parameters:
# sampling rate: 0.0 ~ 1.0 , random select
# model of weak learner: dt （这里面又涉及到 max_depth）, svr （这个参数也挺多的）

# 两种方法

# res_file_path = 'results/method_2_dt%d.csv' % dt_depth
# converge_file_path = 'results/method_converge_dt%d.csv' %dt_depth

res_file_path = 'results/method_2_ada.csv' 
# converge_file_path = 'results/method_converge_ada.csv' %dt_depth

# 这里还需要开另外一个file 来记录模型收敛的情况
def method_2(n_rounds ,  weakLearner_model = 'dt', samplingRate = 0.7, ):
    weak_learner_model = weakLearner_model
    sampling_rate = samplingRate

    # learning starts
    # ---------------------------------- # 
    m = BoostingModel()
    m.clear()
    loss_list = []
    # 这里需要计算残差
    # for each_building in BUILDING_NAMES_MORTAR:
    for round_idx in range(n_rounds):
        each_building = random.choice(BUILDING_NAMES_MORTAR)
        X_train_tmp , y_train_tmp = get_sample_data(X_train_dict[each_building], y_train_dict[each_building], sampling_rate)
        # X_train_tmp = remove_feature(X_train_tmp, REMOVE_FEATURE_INDEX ) # just for experiment
        y_pred_tmp = m.predict(X_train_tmp)
        # 计算残差
        y_residual = BoostingModel.get_residual_error(y_train_tmp, y_pred_tmp)
        # model 拟合残差
        each_model = train_model(X_train_tmp , y_residual, weak_learner_model, max_depth = dt_depth)
        m.add_model(each_model)
        global_loss  = basic_evaluate_mortar(m)        
        loss_list.append(global_loss)

        best_loss = min(loss_list)

        print(best_loss)
        # write_record_to_file(res_str, res_file_path)
        pass
    pass

sampling_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# model_list = ['ada', 'dt']
model_list = ['ada'] 

N_rounds = 50
# model_list = ['nn']

for each_model in model_list:
    for rate in sampling_rate_list:
        method_2(N_rounds,  each_model, rate)
        print(N_rounds, rate, each_model)
        pass
    pass
