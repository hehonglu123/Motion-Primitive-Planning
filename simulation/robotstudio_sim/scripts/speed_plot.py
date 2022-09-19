import numpy as np
import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import read_csv
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *
from MotionSend import *

speed='v55.6640625'
zone='z10'
data_dir='fitting_output/wood/threshold0.2/'
ms=MotionSend()
robot=abb6640(d=50)

###read in curve_exe
df = read_csv(data_dir+"curve_exe"+"_"+speed+"_"+zone+".csv")
df = read_csv('recorded_data/curve_exe_v1000_z10.csv')
lam, curve_exe, curve_exe_R,curve_exe_js, act_speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)
lamdot=calc_lamdot(curve_exe_js,lam,robot,step=1)

plt.plot(lam[1:],act_speed,label='act speed')
# plt.plot(lam,lamdot,label='constraint')
plt.title("Speed: "+data_dir+speed+'_'+zone)
# plt.ylim([0,1600])
plt.ylabel('Speed (mm/s)')
plt.xlabel('lambda (mm)')
plt.show()