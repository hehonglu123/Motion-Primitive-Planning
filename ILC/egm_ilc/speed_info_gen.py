import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *


from robots_def import *
from error_check import *
from MotionSend import *
from utils import *
from lambda_calc import *

ms=MotionSend()
robot=abb6640(d=50)


dataset='wood/'
solution_dir='curve_pose_opt7/'
data_dir="../../data/"+dataset+solution_dir
curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
curve_js = read_csv(data_dir+"Curve_js.csv",header=None).values

exe_dir='recorded_data/'
data=read_csv(exe_dir+"iteration12.csv",header=None).values


idx_delay=50
extension_num=100

##############################data analysis#####################################
timestamp=data[:,0]
curve_exe_js=data[:,1:]

lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp[extension_num+idx_delay:-extension_num+idx_delay],curve_exe_js[extension_num+idx_delay:-extension_num+idx_delay])


error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])

print('average speed',np.average(speed))
print('std speed',np.std(speed))
print('max error',np.max(error))
print('max angle error',np.degrees(np.max(angle_error)))

# df=DataFrame({'average speed':[np.average(speed)],'max speed':[np.amax(speed)],'min speed':[np.amin(speed)],'std speed':[np.std(speed)],\
#     'average error':[np.average(error)],'max error':[np.max(error)],'min error':[np.amin(error)],'std error':[np.std(error)],\
#     'average angle error':[np.average(angle_error)],'max angle error':[max(angle_error)],'min angle error':[np.amin(angle_error)],'std angle error':[np.std(angle_error)]})

# df.to_csv(exe_dir+'speed_info.csv',header=True,index=False)