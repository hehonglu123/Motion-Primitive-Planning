import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from robots_def import *
from error_check import *
from MotionSend import *
from utils import *
from lambda_calc import *
from dual_arm import *

dataset='curve_1/'
data_dir="../../../data/"+dataset
solution_dir=data_dir+'dual_arm/'+'diffevo_pose4/'
exe_dir='../../../ilc/final/recorded_data/'


robot1=robot_obj('ABB_6640_180_255','../../../config/abb_6640_180_255_robot_default_config.yml',tool_file_path='../../../config/paintgun.csv',d=50,acc_dict_path='')
robot2=robot_obj('ABB_1200_5_90','../../../config/abb_1200_5_90_robot_default_config.yml',tool_file_path=solution_dir+'tcp.csv',base_transformation_file=solution_dir+'base.csv',acc_dict_path='')

relative_path,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,data_dir,solution_dir,robot1,robot2)


ms = MotionSend()

data = np.loadtxt(exe_dir+'dual_iteration_10.csv',delimiter=',', skiprows=1)

##############################data analysis#####################################
lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R = ms.logged_data_analysis_multimove(MotionProgramResultLog(None, None, data),robot1,robot2,realrobot=True)
#############################chop extension off##################################
lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
    ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])

##############################calcualte error########################################
error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])


df=DataFrame({'average speed':[np.average(speed)],'max speed':[np.amax(speed)],'min speed':[np.amin(speed)],'std speed':[np.std(speed)],\
    'average error':[np.average(error)],'max error':[np.max(error)],'min error':[np.amin(error)],'std error':[np.std(error)],\
    'average angle error':[np.average(angle_error)],'max angle error':[max(angle_error)],'min angle error':[np.amin(angle_error)],'std angle error':[np.std(angle_error)]})

df.to_csv(exe_dir+'speed_info.csv',header=True,index=False)