import numpy as np
from pandas import *
import sys, traceback, time
from general_robotics_toolbox import *


sys.path.append('../../../toolbox')
from robot_def import *


col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
data = read_csv("relative_path_reverse.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve_direction_x=data['direction_x'].tolist()
curve_direction_y=data['direction_y'].tolist()
curve_direction_z=data['direction_z'].tolist()
relative_path=np.vstack((curve_x, curve_y, curve_z)).T
relative_path_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

relative_path=np.dot(Rx(np.pi),relative_path.T).T
relative_path_direction=np.dot(Rx(np.pi),relative_path_direction.T).T

df=DataFrame({'x':relative_path[:,0],'y':relative_path[:,1], 'z':relative_path[:,2],'x_direction':relative_path_direction[:,0],'y_direction':relative_path_direction[:,1],'z_direction':relative_path_direction[:,2]})
df.to_csv("relative_path_reverse2.csv",header=False,index=False)