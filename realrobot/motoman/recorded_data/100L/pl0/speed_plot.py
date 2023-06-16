import numpy as np
from general_robotics_toolbox import *
import sys
sys.path.append('../../../../toolbox')
from robots_def import *
from error_check import *
from realrobot import *
from MotionSend_motoman import *
from lambda_calc import *

dataset='curve_2/'
solution_dir='baseline_motoman/'
data_dir='../../../../data/'+dataset+solution_dir
cmd_dir=data_dir+'100L/'
curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
lam_original=calc_lam_cs(curve[:,:3])
robot=robot_obj('MA2010_A0',def_path='../../../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../../../config/weldgun2.csv',\
	pulse2deg_file_path='../../../../config/MA2010_A0_pulse2deg.csv',d=50)


ms=MotionSend(robot)

data=np.loadtxt('run_0.csv',delimiter=',')
log_results=(data[:,0],data[:,2:8],data[:,1])
##############################data analysis#####################################
lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,log_results,realrobot=True)
###throw bad curves
lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])



speed_raw=np.sqrt(np.sum(np.diff(curve_exe, axis=0)**2, axis=1)) / np.diff(timestamp)

breakpoints,primitives, p_bp,q_bp=ms.extract_data_from_cmd(cmd_dir+"command.csv")
breakpoints[1:]=breakpoints[1:]-1
for bp in breakpoints:
    plt.axvline(x=lam_original[bp])

plt.plot(lam,speed)
plt.show()