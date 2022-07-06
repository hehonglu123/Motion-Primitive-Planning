import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from io import StringIO

# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *

data_dir='../../../data/wood/'

with open(data_dir+'dual_arm/abb1200.yaml') as file:
    H_1200 = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

base2_R=H_1200[:3,:3]
base2_p=1000*H_1200[:-1,-1]

with open(data_dir+'dual_arm/tcp.yaml') as file:
    H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
robot2=abb1200(R_tool=H_tcp[:3,:3],p_tool=H_tcp[:-1,-1])

ms = MotionSend(robot2=robot2,base2_R=base2_R,base2_p=base2_p)
s=250
v = speeddata(s,9999999,9999999,999999)


cmd_dir='../../../ILC/max_gradient/curve1_250_100L_multipeak/'
relative_path=read_csv(data_dir+"relative_path_tool_frame.csv",header=None).values

#read in initial curve pose
with open(data_dir+'blade_pose.yaml') as file:
	blade_pose = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

curve_js1=read_csv(data_dir+"Curve_js.csv",header=None).values




q_init1=curve_js1[0]
q_init2=ms.calc_robot2_q_from_blade_pose(blade_pose,base2_R,base2_p)
p_init2=robot2.fwd(q_init2).p
   
breakpoints1,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(cmd_dir+'command.csv')
breakpoints2=copy.deepcopy(breakpoints1)
primitives2=['movej_fit']*len(primitives1)
p_bp2=[[p_init2]]*len(primitives1)
q_bp2=[[q_init2]]*len(primitives1)

logged_data=ms.exec_motions_multimove(breakpoints1,primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,v,v,z10,z10)

StringData=StringIO(logged_data)
df = read_csv(StringData, sep =",")
##############################data analysis#####################################
lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R = ms.logged_data_analysis_multimove(df,base2_R,base2_p,realrobot=True)
#############################chop extension off##################################
lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
	ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])

##############################calcualte error########################################
error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])
print(max(error))

##############################plot error#####################################

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(lam, speed, 'g-', label='Speed')
ax2.plot(lam, error, 'b-',label='Error')
ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
ax2.axis(ymin=0,ymax=4)

ax1.set_xlabel('lambda (mm)')
ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
plt.title("Speed and Error Plot")
ax1.legend(loc=0)

ax2.legend(loc=0)

plt.legend()
plt.show()