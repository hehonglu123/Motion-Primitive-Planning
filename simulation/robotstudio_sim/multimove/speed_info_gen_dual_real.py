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
from realrobot import *

dataset='curve_1/'
data_dir="../../../data/"+dataset
solution_dir=data_dir+'dual_arm/'+'diffevo_pose4/'
exe_dir='../../../ilc/final/curve1_dual_v400_real/'


robot1=robot_obj('ABB_6640_180_255','../../../config/abb_6640_180_255_robot_default_config.yml',tool_file_path='../../../config/paintgun.csv',d=50,acc_dict_path='')
robot2=robot_obj('ABB_1200_5_90','../../../config/abb_1200_5_90_robot_default_config.yml',tool_file_path=solution_dir+'tcp.csv',base_transformation_file=solution_dir+'base.csv',acc_dict_path='')

relative_path,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,data_dir,solution_dir,robot1,robot2)


ms = MotionSend()

curve_exe_js_all=[]
timestamp_all=[]
total_time_all=[]

for i in range(10):
    data = np.loadtxt(exe_dir+'run_'+str(i)+'.csv',delimiter=',', skiprows=1)
    ##############################data analysis#####################################
    lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=ms.logged_data_analysis_multimove(MotionProgramResultLog(None, None, data),robot1,robot2,realrobot=True)
    ###throw bad curves
    _, _,_,_,_,_,_, _, timestamp_temp, _, _=\
        ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])

    curve_exe_js_dual=np.hstack((curve_exe_js1,curve_exe_js2))
    total_time_all.append(timestamp_temp[-1]-timestamp_temp[0])

    timestamp=timestamp-timestamp[0]

    curve_exe_js_all.append(curve_exe_js_dual)
    timestamp_all.append(timestamp)


###trajectory outlier detection, based on chopped time
curve_exe_js_all,timestamp_all=remove_traj_outlier(curve_exe_js_all,timestamp_all,total_time_all)

###infer average curve from linear interplateion
curve_js_all_new, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)

###calculat data with average curve
lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R =\
    logged_data_analysis_multimove(robot1,robot2,timestamp_d,avg_curve_js)

#############################chop extension off##################################
lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
    ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])


##############################calcualte error########################################
error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])


df=DataFrame({'average speed':[np.average(speed)],'max speed':[np.amax(speed)],'min speed':[np.amin(speed)],'std speed':[np.std(speed)],\
    'average error':[np.average(error)],'max error':[np.max(error)],'min error':[np.amin(error)],'std error':[np.std(error)],\
    'average angle error':[np.average(angle_error)],'max angle error':[max(angle_error)],'min angle error':[np.amin(angle_error)],'std angle error':[np.std(angle_error)]})

df.to_csv(exe_dir+'speed_info.csv',header=True,index=False)