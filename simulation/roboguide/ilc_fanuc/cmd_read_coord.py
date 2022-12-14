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

# curve
data_type='curve_1'
# data_type='curve_2_scale'

# data and curve directory
curve_data_dir='../../../data/'+data_type+'/'

# test_type='50L'
# test_type='30L'
# test_type='greedy0.2'
# test_type='greedy0.02'
# test_type='moveLgreedy0.2'
# test_type='moveLgreedy0.02'
test_type='minStepMoveLgreedy0.2'
# test_type='minStepgreedy0.02'
# test_type='minStepgreedy0.2'

cmd_dir='../data/'+data_type+'/dual_arm_de/'+test_type+'/'
# cmd_dir='../data/'+data_type+'/dual_arm_de_possibilyimpossible/'+test_type+'/'

# relative path
relative_path = read_csv(curve_data_dir+"/Curve_dense.csv", header=None).values

H_200id=np.loadtxt(cmd_dir+'../base.csv',delimiter=',')
base2_R=H_200id[:3,:3]
base2_p=H_200id[:-1,-1]
# print(base2_R)
print(base2_p)
k,theta=R2rot(base2_R)
print(k,np.degrees(theta))

robot2_tcp=np.loadtxt(cmd_dir+'../tcp.csv',delimiter=',')
# print(robot2_tcp)

# exit()

## robot
toolbox_path = '../../../toolbox/'
robot1 = robot_obj('FANUC_m10ia',toolbox_path+'robot_info/fanuc_m10ia_robot_default_config.yml',tool_file_path=toolbox_path+'tool_info/paintgun.csv',d=50,acc_dict_path=toolbox_path+'robot_info/m10ia_acc_compensate.pickle',j_compensation=[1,1,-1,-1,-1,-1])
robot2=robot_obj('FANUC_lrmate200id',toolbox_path+'robot_info/fanuc_lrmate200id_robot_default_config.yml',tool_file_path=cmd_dir+'../tcp.csv',base_transformation_file=cmd_dir+'../base.csv',acc_dict_path=toolbox_path+'robot_info/lrmate200id_acc_compensate.pickle',j_compensation=[1,1,-1,-1,-1,-1])

# robot2=lrmate200id(R_tool=Ry(np.pi/2)@robot2_tcp[:3,:3],p_tool=Ry(np.pi/2)@robot2_tcp[:3,-1])
# print(robot2.fwd(np.radians([-41.52,-12.97,156.93,-57.35,-31.39,97.72])))
# print(robot2.fwd(np.radians([0,0,0,0,0,0])))
# exit()

# fanuc motion send tool
if data_type=='curve_1':
    ms = MotionSendFANUC(robot1=robot1,robot2=robot2)
elif data_type=='curve_2_scale':
    ms = MotionSendFANUC(robot1=robot1,robot2=robot2,utool2=3)

s=500 # mm/sec in leader frame
z=100 # CNT100

iteration=2
ilc_output=cmd_dir+'results_'+str(s)+'_'+test_type+'/'
try:
    breakpoints1,primitives1,p_bp1,q_bp1,_=ms.extract_data_from_cmd(os.getcwd()+'/'+ilc_output+'command_arm1_'+str(iteration)+'.csv')
    breakpoints2,primitives2,p_bp2,q_bp2,_=ms.extract_data_from_cmd(os.getcwd()+'/'+ilc_output+'command_arm2_'+str(iteration)+'.csv')
except Exception as e:
    print(e)
    print("Convert bp to command")
    exit()


###execution with plant
logged_data=ms.exec_motions_multimove(robot1,robot2,primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,s,z)
# with open('iteration_'+str(i)+'.csv',"wb") as f:
#     f.write(logged_data)
StringData=StringIO(logged_data.decode('utf-8'))
df = read_csv(StringData, sep =",")
##############################data analysis#####################################
lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R = ms.logged_data_analysis_multimove(df,base2_R,base2_p,realrobot=False)
#############################chop extension off##################################
lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
    ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])
ave_speed=np.mean(speed)

##############################calcualte error########################################
error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])
print('Iteration:',iteration,', Max Error:',max(error),'Ave. Speed:',ave_speed,'Std. Speed:',np.std(speed),'Std/Ave (%):',np.std(speed)/ave_speed*100)
print('Max Speed:',max(speed),'Min Speed:',np.min(speed),'Ave. Error:',np.mean(error),'Min Error:',np.min(error),"Std. Error:",np.std(error))
print('Max Ang Error:',max(np.degrees(angle_error)),'Min Ang Error:',np.min(np.degrees(angle_error)),'Ave. Ang Error:',np.mean(np.degrees(angle_error)),"Std. Ang Error:",np.std(np.degrees(angle_error)))
print("===========================================")

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