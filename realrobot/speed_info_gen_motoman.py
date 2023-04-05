import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys

sys.path.append('../toolbox')
from robots_def import *
from error_check import *
from MotionSend_motoman import *
from utils import *
from lambda_calc import *
from realrobot import *

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun2.csv',\
    pulse2deg_file_path='../config/MA2010_A0_pulse2deg.csv',d=50)
ms=MotionSend(robot)


dataset='curve_2/'

solution_dir='baseline_motoman/'
cmd_dir='../data/'+dataset+solution_dir+'200L/'
data_dir='../data/'+dataset+solution_dir

curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values

exe_dir=cmd_dir+'iteration_7/'

N=5
curve_exe_js_all=[]
timestamp_all=[]
total_time_all=[]
for i in range(N):
    data=np.loadtxt(exe_dir+'run_'+str(i)+'.csv',delimiter=',')
    log_results=(data[:,0],data[:,2:8],data[:,1])
    ##############################data analysis#####################################
    lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,log_results,realrobot=True)

    ###throw bad curves
    lam_temp, curve_exe_temp, curve_exe_R_temp,curve_exe_js_temp, speed_temp, timestamp_temp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])
    error,angle_error=calc_all_error_w_normal(curve_exe_temp,curve[:,:3],curve_exe_R_temp[:,:,-1],curve[:,3:])
    total_time_all.append(timestamp_temp[-1]-timestamp_temp[0])
    timestamp=timestamp-timestamp[0]
    curve_exe_js_all.append(curve_exe_js)
    timestamp_all.append(timestamp)

###trajectory outlier detection, based on chopped time
curve_exe_js_all,timestamp_all=remove_traj_outlier(curve_exe_js_all,timestamp_all,total_time_all)

###infer average curve from linear interplateion
curve_js_all_new, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)

lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp_d,avg_curve_js)
#############################chop extension off##################################
lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,avg_curve_js, speed, timestamp_d,curve[0,:3],curve[-1,:3])

error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])




df=DataFrame({'average speed':[np.average(speed)],'max speed':[np.amax(speed)],'min speed':[np.amin(speed)],'std speed':[np.std(speed)],\
    'average error':[np.average(error)],'max error':[np.max(error)],'min error':[np.amin(error)],'std error':[np.std(error)],\
    'average angle error':[np.average(angle_error)],'max angle error':[max(angle_error)],'min angle error':[np.amin(angle_error)],'std angle error':[np.std(angle_error)]})

df.to_csv(exe_dir+'speed_info.csv',header=True,index=False)