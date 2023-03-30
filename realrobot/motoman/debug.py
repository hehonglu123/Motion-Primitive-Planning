import numpy as np
from general_robotics_toolbox import *
import sys
sys.path.append('../../toolbox')
from robots_def import *
from error_check import *
from realrobot import *
from MotionSend_motoman import *
from lambda_calc import *

dataset='curve_1/'
solution_dir='baseline_motoman/'
data_dir='../../data/'+dataset+solution_dir
cmd_dir=data_dir+'100L/'


robot=robot_obj('MA2010_A0',def_path='../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun2.csv',\
    pulse2deg_file_path='../../config/MA2010_A0_pulse2deg.csv',d=50)

recorded_dir='recorded_data/iteration_1/'

###N run execute
N=2
curve_exe_js_all=[]
timestamp_all=[]
total_time_all=[]
for i in range(N):
	data=np.loadtxt(recorded_dir+'run_'+str(i)+'.csv',delimiter=',')
	log_results=(data[:,0],data[:,1:7])
	##############################data analysis#####################################
	lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,log_results,realrobot=True)

	###throw bad curves
	_, _, _,_, _, timestamp_temp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])

	total_time_all.append(timestamp_temp[-1]-timestamp_temp[0])
	timestamp=timestamp-timestamp[0]
	curve_exe_js_all.append(curve_exe_js)
	timestamp_all.append(timestamp)



print(total_time_all)
###trajectory outlier detection, based on chopped time
curve_exe_js_all,timestamp_all=remove_traj_outlier(curve_exe_js_all,timestamp_all,total_time_all)