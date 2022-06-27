import numpy as np
import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import read_csv

# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *
from MotionSend import *

dataset='from_NX/'
data_dir="../data/"+dataset
fitting_output="../data/"+dataset+'baseline/100L/'
curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values

ms=MotionSend()
robot=abb6640(d=50)

exe_dir='recorded_data/iteration_2/'
###5 run execute
curve_exe_all=[]
curve_exe_js_all=[]
timestamp_all=[]
total_time_all=[]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
###read in curve_exe
for i in range(5):
    df = read_csv(exe_dir+"run"+"_"+str(i)+".csv")
    lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)
    
    lam_temp, curve_exe_temp, curve_exe_R_temp,curve_exe_js_temp, speed_temp, timestamp_temp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])

    total_time_all.append(timestamp_temp[-1]-timestamp_temp[0])

    curve_exe_all.append(curve_exe)
    curve_exe_js_all.append(curve_exe_js)
    timestamp_all.append(timestamp)

###trajectory outlier detection, based on chopped time
curve_exe_all,curve_exe_js_all,timestamp_all=remove_traj_outlier(curve_exe_all,curve_exe_js_all,timestamp_all,total_time_all)



###infer average curve from linear interplateion
curve_js_all_new, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)

# #############################chop extension off##################################
lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp_d,curve[0,:3],curve[-1,:3])


# ##############################calcualte error########################################
error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])

print('speed standard deviation: ',np.std(speed))
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(lam,speed, 'g-', label='Speed')
ax2.plot(lam, error, 'b-',label='Error')
ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')

ax1.set_xlabel('lambda (mm)')
ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
ax1.legend(loc=0)

ax2.legend(loc=0)

plt.legend()
plt.show()
