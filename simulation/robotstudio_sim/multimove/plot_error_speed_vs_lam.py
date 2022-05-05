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
data = read_csv("../../../data/from_ge/relative_path_tool_frame.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve=np.vstack((curve_x, curve_y, curve_z)).T

speed=['vmax']
zone=['z10']
data_dir=""

for s in speed:
    for z in zone:
        ###read in curve_exe
        col_names=['timestamp', 'cmd_num', 'J1', 'J2','J3', 'J4', 'J5', 'J6','J1_2', 'J2_2','J3_2', 'J4_2', 'J5_2', 'J6_2'] 
        data = read_csv(data_dir+"curve_exe_"+s+'_'+z+".csv",names=col_names)
        q1_1=data['J1'].tolist()[1:]
        q1_2=data['J2'].tolist()[1:]
        q1_3=data['J3'].tolist()[1:]
        q1_4=data['J4'].tolist()[1:]
        q1_5=data['J5'].tolist()[1:]
        q1_6=data['J6'].tolist()[1:]
        q2_1=data['J1_2'].tolist()[1:]
        q2_2=data['J2_2'].tolist()[1:]
        q2_3=data['J3_2'].tolist()[1:]
        q2_4=data['J4_2'].tolist()[1:]
        q2_5=data['J5_2'].tolist()[1:]
        q2_6=data['J6_2'].tolist()[1:]

        cmd_num=np.array(data['cmd_num'].tolist()[1:]).astype(float)
        start_idx=np.where(cmd_num==5)[0][0]
        curve_exe_js1=np.radians(np.vstack((q1_1,q1_2,q1_3,q1_4,q1_5,q1_6)).T.astype(float)[start_idx:])
        curve_exe_js2=np.radians(np.vstack((q2_1,q2_2,q2_3,q2_4,q2_5,q2_6)).T.astype(float)[start_idx:])
        timestamp=np.array(data['timestamp'].tolist()[start_idx:]).astype(float)

        timestep=np.average(timestamp[1:]-timestamp[:-1])

        robot1=abb1200(d=50)
        robot2=abb6640()
        base2_R=np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        base2_p=np.array([3000,1000,0])

        act_speed=[]
        lam=[0]
        curve_exe=[]
        curve_exe_R=[]
        for i in range(len(curve_exe_js1)):
            pose1_now=robot1.fwd(curve_exe_js1[i])
            pose2_now=robot2.fwd(curve_exe_js2[i])

            pose2_world_now=robot2.fwd(curve_exe_js2[i],base2_R,base2_p)


            curve_exe.append(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p))
            curve_exe_R.append(pose2_world_now.R.T@pose1_now.R)
            if i>0:
                lam.append(lam[-1]+np.linalg.norm(curve_exe[i]-curve_exe[i-1]))
            try:
                if timestamp[i-1]!=timestamp[i] and np.linalg.norm(curve_exe[-1]-curve_exe[-2])!=0:
                    act_speed.append(np.linalg.norm(curve_exe[-1]-curve_exe[-2])/(timestamp[i]-timestamp[i-1]))
                else:
                    act_speed.append(act_speed[-1])
                    
            except IndexError:
                pass

        error=calc_all_error(curve_exe,curve)

        dlam=calc_lamdot_2arm(np.hstack((curve_exe_js1,curve_exe_js2)),lam,robot1,robot2,step=1)



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
