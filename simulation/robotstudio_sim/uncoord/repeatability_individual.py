import numpy as np
import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import read_csv
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *
from MotionSend import *

ms = MotionSend()
robot1=abb6640()
robot2=abb1200()

df = read_csv('recorded_data/log1_1.csv')
_, _, _,curve_exe_js_it0, _, timestamp_it0=ms.logged_data_analysis(robot1,df,realrobot=True)

for i in range(1,5):
    df = read_csv('recorded_data/log1_'+str(i+1)+'.csv')
    _, _, _,curve_exe_js_temp, _, timestamp_temp=ms.logged_data_analysis(robot1,df,realrobot=True)
    error_temp=curve_exe_js_temp-curve_exe_js_it0
    plt.plot(timestamp_it0,np.linalg.norm(error_temp,axis=1))

plt.xlabel('time (s)')
plt.ylabel('error norm (rad)')
plt.title('robot1 joint repeatibility')
plt.show()

df = read_csv('recorded_data/log2_1.csv')
_, _, _,curve_exe_js_it0, _, timestamp_it0=ms.logged_data_analysis(robot2,df,realrobot=True)

for i in range(1,5):
    df = read_csv('recorded_data/log2_'+str(i+1)+'.csv')
    _, _, _,curve_exe_js_temp, _, timestamp_temp=ms.logged_data_analysis(robot2,df,realrobot=True)
    error_temp=curve_exe_js_temp-curve_exe_js_it0
    plt.plot(timestamp_it0,np.linalg.norm(error_temp,axis=1))

plt.xlabel('time (s)')
plt.ylabel('error norm (rad)')
plt.title('robot2 joint repeatibility')
plt.show()