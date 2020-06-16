from utils.building_data_IO import BUILDING_NAMES_MORTAR
from utils.building_data_IO import prepare_dataset


for building_name in BUILDING_NAMES_MORTAR:
    X, y = prepare_dataset('Mortar', building_name)
    pass



