import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *


col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
data = read_csv("../../../data/from_ge/Curve_backproj_in_base_frame.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve=np.vstack((curve_x, curve_y, curve_z)).T

col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
data = read_csv("curve_fit_backproj.csv")
curve_x=data['x'].tolist()
curve_y=data['y'].tolist()
curve_z=data['z'].tolist()
curve_fit=np.vstack((curve_x, curve_y, curve_z)).T


###read in curve_exe
col_names=['timestamp', 'cmd_num', 'J1', 'J2','J3', 'J4', 'J5', 'J6'] 
data = read_csv("curve_exe.csv",names=col_names)
q1=data['J1'].tolist()[1:]
q2=data['J2'].tolist()[1:]
q3=data['J3'].tolist()[1:]
q4=data['J4'].tolist()[1:]
q5=data['J5'].tolist()[1:]
q6=data['J6'].tolist()[1:]
timestamp=data['timestamp'].tolist()[1:]
cmd_num=np.array(data['cmd_num'].tolist()[1:]).astype(float)
start_idx=np.where(cmd_num==2)[0][0]
curve_exe_js=np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)[start_idx:]


robot=abb6640()
curve_exe=[]
for i in range(len(curve_exe_js)):
    curve_exe.append(robot.fwd(np.radians(curve_exe_js[i])).p)
print('max error: ',calc_max_error(curve_exe,curve_fit))