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

# curve = read_csv('../../../data/wood/Curve_in_base_frame.csv',header=None).values
curve = read_csv('../../../data/from_ge/Curve_in_base_frame2.csv',header=None).values

speed=['v500']
zone=['z10']
data_dir="tesseract/"

ms=MotionSend()
robot=abb6640(d=50)
for s in speed:
    for z in zone:
        ###read in curve_exe
        df = read_csv(data_dir+"curve_exe"+"_"+s+"_"+z+".csv")
        lam, curve_exe, curve_exe_R,curve_exe_js, act_speed, timestamp=ms.logged_data_analysis(robot,df)

        error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])

        print('speed standard deviation: ',np.std(act_speed))
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(lam[1:],act_speed, 'g-', label='Speed')
        ax2.plot(lam, error, 'b-',label='Error')
        ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')

        ax1.set_xlabel('lambda (mm)')
        ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
        ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
        plt.title("Speed: "+data_dir+s+'_'+z)
        ax1.legend(loc=0)

        ax2.legend(loc=0)

        plt.legend()
        plt.show()
