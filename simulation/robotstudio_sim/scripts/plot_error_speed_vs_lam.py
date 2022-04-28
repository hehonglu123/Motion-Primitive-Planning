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
from scipy.interpolate import UnivariateSpline

col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
data = read_csv("../../../data/from_ge/Curve_in_base_frame2.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve=np.vstack((curve_x, curve_y, curve_z)).T

speed=['v150','v200','v400','v800']
zone=['z10']
data_dir="fitting_output_new/matlab_qp_movel/"

for s in speed:
    for z in zone:
        ###read in curve_exe
        col_names=['timestamp', 'cmd_num', 'J1', 'J2','J3', 'J4', 'J5', 'J6'] 
        data = read_csv(data_dir+"curve_exe_"+s+'_'+z+".csv",names=col_names)
        q1=data['J1'].tolist()[1:]
        q2=data['J2'].tolist()[1:]
        q3=data['J3'].tolist()[1:]
        q4=data['J4'].tolist()[1:]
        q5=data['J5'].tolist()[1:]
        q6=data['J6'].tolist()[1:]
        cmd_num=np.array(data['cmd_num'].tolist()[1:]).astype(float)
        start_idx=np.where(cmd_num==6)[0][0]
        curve_exe_js=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)[start_idx:])
        timestamp=np.array(data['timestamp'].tolist()[start_idx:]).astype(float)

        timestep=np.average(timestamp[1:]-timestamp[:-1])

        robot=abb6640(d=50)
        act_speed=[]
        lam=[0]
        curve_exe=[]
        curve_exe_R=[]
        for i in range(len(curve_exe_js)):
            robot_pose=robot.fwd(curve_exe_js[i])
            curve_exe.append(robot_pose.p)
            curve_exe_R.append(robot_pose.R)
            if i>0:
                lam.append(lam[-1]+np.linalg.norm(curve_exe[i]-curve_exe[i-1]))
            try:
                if timestamp[-1]!=timestamp[-2]:
                    act_speed.append(np.linalg.norm(curve_exe[-1]-curve_exe[-2])/timestep)
                    
            except IndexError:
                pass

        error=calc_all_error(curve_exe,curve)

        lamdot_act=calc_lamdot(curve_exe_js,lam,robot,1)


        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(lam[1:],act_speed, 'g-', label='Speed')
        ax2.plot(lam, error, 'b-',label='Error')
        # ax1.plot(lam[2:],lamdot_act[2:], 'r-',label='Lamdot Constraint')

        ax1.set_xlabel('lambda (mm)')
        ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
        ax2.set_ylabel('Error (mm)', color='b')
        plt.title("Speed: "+data_dir+s+'_'+z)
        ax1.legend(loc=0)

        ax2.legend(loc=0)

        plt.legend()
        plt.show()
