import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys

sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from utils import *
from lambda_calc import *
from realrobot import *

N=5

ms=MotionSend()
robot=robot_obj('ABB_6640_180_255','../../../config/abb_6640_180_255_robot_default_config.yml',tool_file_path='../../../config/paintgun.csv',d=50,acc_dict_path='')

dataset='curve_2/'
solution_dir='curve_pose_opt2/'
data_dir="../../../data/"+dataset+solution_dir

curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values

exe_dir='../../../ILC/final/recorded_data/'

curve_exe_js_all=[]
timestamp_all=[]
total_time_all=[]
for i in range(N):
    data = np.loadtxt(exe_dir+'run_'+str(i)+'.csv',delimiter=',', skiprows=1)
    ##############################data analysis#####################################
    lam, curve_exe, curve_exe_R, curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,MotionProgramResultLog(None, None, data),realrobot=True)
    ###throw bad curves
    _, _, _,_, _, timestamp_temp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])

    total_time_all.append(timestamp_temp[-1]-timestamp_temp[0])
    timestamp=timestamp-timestamp[0]
    curve_exe_js_all.append(curve_exe_js)
    timestamp_all.append(timestamp)

###trajectory outlier detection, based on chopped time
curve_exe_js_all,timestamp_all=remove_traj_outlier(curve_exe_js_all,timestamp_all,total_time_all)

###infer average curve from linear interplateion
_, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)

##############################data analysis#####################################
lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp_d,avg_curve_js)
lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,avg_curve_js, speed, timestamp_d,curve[0,:3],curve[-1,:3])
error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])

error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])

start_idx=np.argmin(np.linalg.norm(curve[0,:3]-curve_exe,axis=1))
end_idx=np.argmin(np.linalg.norm(curve[-1,:3]-curve_exe,axis=1))

curve_exe=curve_exe[start_idx:end_idx+1]
curve_exe_R=curve_exe_R[start_idx:end_idx+1]
speed=speed[start_idx:end_idx+1]
lam=calc_lam_cs(curve_exe)

speed=replace_outliers(np.array(speed))


speed[start_idx:end_idx+1]
df=DataFrame({'average speed':[np.average(speed)],'max speed':[np.amax(speed)],'min speed':[np.amin(speed)],'std speed':[np.std(speed)],\
    'average error':[np.average(error)],'max error':[np.max(error)],'min error':[np.amin(error)],'std error':[np.std(error)],\
    'average angle error':[np.average(angle_error)],'max angle error':[max(angle_error)],'min angle error':[np.amin(angle_error)],'std angle error':[np.std(angle_error)]})

df.to_csv(exe_dir+'speed_info.csv',header=True,index=False)