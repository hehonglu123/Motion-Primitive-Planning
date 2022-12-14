import numpy as np
from math import degrees,radians,ceil,floor
import yaml
from copy import deepcopy
from scipy.signal import find_peaks
from general_robotics_toolbox import *
from pandas import read_csv,DataFrame
import sys
from io import StringIO
from matplotlib import pyplot as plt
from pathlib import Path
import os
from fanuc_motion_program_exec_client import *

# sys.path.append('../abb_motion_program_exec')
# from abb_motion_program_exec_client import *
sys.path.append('../fanuc_toolbox')
from fanuc_utils import *
sys.path.append('../../../ILC')
from ilc_toolbox import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *
from blending import *

# obj_type='curve_1'
obj_type='curve_2_scale'
data_dir='../data/'+obj_type+'/single_arm_de/'

robot=m10ia(d=50)
curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
curve = np.array(curve)
curve_normal = curve[:,3:]
curve = curve[:,:3]

multi_peak_threshold=0.2
robot=m10ia(d=50)
ms = MotionSendFANUC()

num_l=25
s=727
z=100
iteration=2
# cmd_dir='../data/baseline_m10ia/'+obj_type+'/'+str(num_l)+'/'+'results_'+str(s)+'/'
cmd_dir=data_dir+'greedy_20/results_727/'
try:
    breakpoints,primitives,p_bp,q_bp,_=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command_arm1_'+str(iteration)+'.csv')
    # breakpoints,primitives,p_bp,q_bp=extract_data_from_cmd(os.getcwd()+'/'+ilc_output+'command_25.csv')
except Exception as e:
    print(e)
    print("Convert bp to command")
    exit()


###execute,curve_fit_js only used for orientation
logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,s,z)
StringData=StringIO(logged_data.decode('utf-8'))
df = read_csv(StringData)
##############################data analysis#####################################
lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df)

#############################chop extension off##################################
lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[:,:3],curve_normal)
ave_speed=np.mean(speed)

##############################calcualte error########################################
error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve_normal)
print('Iteration:',iteration,', Max Error:',max(error),'Ave. Speed:',ave_speed,'Std. Speed:',np.std(speed),'Std/Ave (%):',np.std(speed)/ave_speed*100)
print('Max Speed:',max(speed),'Min Speed:',np.min(speed),'Ave. Error:',np.mean(error),'Min Error:',np.min(error),"Std. Error:",np.std(error))
print('Max Ang Error:',max(np.degrees(angle_error)),'Min Ang Error:',np.min(np.degrees(angle_error)),'Ave. Ang Error:',np.mean(np.degrees(angle_error)),"Std. Ang Error:",np.std(np.degrees(angle_error)))
print("===========================================")

#############################error peak detection###############################
find_peak_dist = 20/(lam[int(len(lam)/2)]-lam[int(len(lam)/2)-1])
if find_peak_dist<1:
    find_peak_dist=1
peaks,_=find_peaks(error,height=multi_peak_threshold,prominence=0.05,distance=find_peak_dist)		###only push down peaks higher than height, distance between each peak is 20mm, threshold to filter noisy peaks

if len(peaks)==0 or np.argmax(error) not in peaks:
    peaks=np.append(peaks,np.argmax(error))

##############################plot error#####################################
fig, ax1 = plt.subplots(figsize=(6,4))
ax2 = ax1.twinx()
ax1.plot(lam, speed, 'g-', label='Speed')
ax2.plot(lam, error, 'b-',label='Error')
ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
draw_speed_max=max(speed)*1.05
ax1.axis(ymin=0,ymax=draw_speed_max)
draw_error_max=max(np.append(error,np.degrees(angle_error)))*1.05
ax2.axis(ymin=0,ymax=draw_error_max)

ax1.set_xlabel('lambda (mm)')
ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
plt.title("Speed and Error Plot")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc=1)

plt.show()