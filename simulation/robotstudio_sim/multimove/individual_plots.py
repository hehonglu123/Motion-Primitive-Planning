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


#kin def
robot1=abb1200(d=50)
robot2=abb6640()
base2_R=np.array([[-1,0,0],[0,-1,0],[0,0,1]])
base2_p=np.array([3000,1000,0])

data_dir='wood/greedy_dual_output/'
data1 = read_csv(data_dir+"curve_fit1.csv")
data2 = read_csv(data_dir+"curve_fit2.csv")

curve_fit1=np.vstack((data1['x'].values,data1['y'].values,data1['z'].values)).T
curve_fit2=np.vstack((data2['x'].values,data2['y'].values,data2['z'].values)).T


speed=['v50']
zone=['z1']
data_dir="wood/greedy_dual_output/"

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

        
        
        
        curve_exe1=[]
        curve_exe2=[]
        relative_path_exe=[]
        for i in range(len(curve_exe_js1)):
            pose1_now=robot1.fwd(curve_exe_js1[i])
            pose2_now=robot2.fwd(curve_exe_js2[i])

            pose2_world_now=robot2.fwd(curve_exe_js2[i],base2_R,base2_p)

            curve_exe1.append(pose1_now.p)
            curve_exe2.append(pose2_now.p)

            relative_path_exe.append(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p))

        curve_exe1=np.array(curve_exe1)
        curve_exe2=np.array(curve_exe2)
        relative_path_exe=np.array(relative_path_exe)


        error1=calc_all_error(curve_exe1,curve_fit1)
        error2=calc_all_error(curve_exe2,curve_fit2)

        lam=calc_lam_cs(relative_path_exe)

        plt.figure()
        plt.plot(lam,error1,label='error1')
        plt.plot(lam,error2,label='error2')
        plt.xlabel('lambda (mm)')
        plt.ylabel('error (mm)')
        plt.title("Error: "+data_dir+s+'_'+z)
        plt.legend()
        # plt.show()

        ###3D plot in global frame
        curve_fit2_global=(base2_R@curve_fit2.T).T+np.tile(base2_p,(len(curve_fit2),1))
        curve_exe2_global=(base2_R@curve_exe2.T).T+np.tile(base2_p,(len(curve_exe2),1))
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(curve_fit1[:,0], curve_fit1[:,1],curve_fit1[:,2], 'gray', label='arm1_fit')
        ax.plot3D(curve_fit2_global[:,0], curve_fit2_global[:,1],curve_fit2_global[:,2], 'green', label='arm2_fit')
        ax.plot3D(curve_exe1[:,0], curve_exe1[:,1],curve_exe1[:,2], 'red', label='arm1_exe')
        ax.plot3D(curve_exe2_global[:,0], curve_exe2_global[:,1],curve_exe2_global[:,2], 'blue', label='arm2_exe')
        plt.legend()
        plt.show()