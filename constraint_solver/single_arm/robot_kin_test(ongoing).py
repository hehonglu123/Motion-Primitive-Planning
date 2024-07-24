import sys
sys.path.append('../../toolbox')
from robot_def import *
sys.path.append('../')
from constraint_solver import *

robot=robot_obj('ABB_6640_180_255','../../config/ABB_6640_180_255_robot_default_config.yml',tool_file_path='../../config/paintgun.csv',d=50,acc_dict_path='../../config/acceleration/6640acc_new.pickle')

q=np.ones(6)
print(robot.fwd(q).p)

