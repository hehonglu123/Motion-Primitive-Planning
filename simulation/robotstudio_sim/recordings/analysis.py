import numpy as np
import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import read_csv, read_excel
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *


###read in curve_exe
data = read_excel("Recording-1.xlsx")
q1=data['J1'].tolist()[1:]
q2=data['J2'].tolist()[1:]
q3=data['J3'].tolist()[1:]
q4=data['J4'].tolist()[1:]
q5=data['J5'].tolist()[1:]
q6=data['J6'].tolist()[1:]

timestamp=np.array(data['Time'].tolist()[1:]).astype(float)
speed=np.array(data['Speed'].tolist()[1:]).astype(float)
start_idx=np.where(speed==0)[0][0]
curve_exe_js=np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)[start_idx:]


robot=abb6640()
act_speed=[]
lam=[0]
curve_exe=[]
curve_exe_R=[]
for i in range(len(curve_exe_js)):
    robot_pose=robot.fwd(np.radians(curve_exe_js[i]))
    curve_exe.append(robot_pose.p)
    curve_exe_R.append(robot_pose.R)
    if i>0:
        lam.append(lam[-1]+np.linalg.norm(curve_exe[i]-curve_exe[i-1]))
    try:
        if timestamp[-1]!=timestamp[-2]:
            act_speed.append(np.linalg.norm(curve_exe[-1]-curve_exe[-2])/(timestamp[-1]- timestamp[-2]))
            
    except IndexError:
        pass

plt.plot(lam[1:],act_speed)
plt.title("Speed")
plt.ylabel('Speed (mm/s)')
plt.xlabel('lambda (mm)')
plt.show()