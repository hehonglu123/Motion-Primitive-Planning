import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from utils import *
from lambda_calc import *

ms=MotionSend()
robot=abb6640(d=50)


dataset='from_NX/'
data_dir="../data/"+dataset
curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values

# exe_dir='../ILC/max_gradient/curve1_250_100L_multipeak_real/'
exe_dir='../ILC/max_gradient/curve2_1100_100L_multipeak_real/'

###5 run execute
curve_exe_all=[]
curve_exe_js_all=[]
timestamp_all=[]
total_time_all=[]
###clear globals
lam_temp_all=[]
error_temp_all=[]
speed_temp_all=[]
angle_error_temp_all=[]
error=[]
##############################data analysis#####################################
for r in range(5):
    df = read_csv(exe_dir+'run_'+str(r)+'.csv')
    lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)
    ###throw bad curves, error calc for individual traj for demo
    lam_temp, curve_exe_temp, curve_exe_R_temp,curve_exe_js_temp, speed_temp, timestamp_temp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])
    error_temp,angle_error_temp=calc_all_error_w_normal(curve_exe_temp,curve[:,:3],curve_exe_R_temp[:,:,-1],curve[:,3:])
    lam_temp_all.append(lam_temp)
    error_temp_all.append(error_temp)
    speed_temp_all.append(speed_temp)
    angle_error_temp_all.append(angle_error_temp)

    total_time_all.append(timestamp_temp[-1]-timestamp_temp[0])

    timestamp=timestamp-timestamp[0]

    curve_exe_all.append(curve_exe)
    curve_exe_js_all.append(curve_exe_js)
    timestamp_all.append(timestamp)

###trajectory outlier detection, based on chopped time
curve_exe_all,curve_exe_js_all,timestamp_all=remove_traj_outlier(curve_exe_all,curve_exe_js_all,timestamp_all,total_time_all)

###infer average curve from linear interplateion
curve_js_all_new, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)
###calculat data with average curve
lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp_d,avg_curve_js)
#############################chop extension off##################################
lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp_d,curve[0,:3],curve[-1,:3])


##############################calcualte error########################################
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