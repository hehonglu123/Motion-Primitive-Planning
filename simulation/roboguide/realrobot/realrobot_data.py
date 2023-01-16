import numpy as np
from math import degrees,radians,ceil,floor
import yaml
from copy import deepcopy
from scipy.signal import find_peaks
from general_robotics_toolbox import *
from pandas import read_csv,DataFrame
import sys
from io import StringIO
from matplotlib import pyplot as plt
from pathlib import Path
import os
from fanuc_motion_program_exec_client import *

# sys.path.append('../abb_motion_program_exec')
# from abb_motion_program_exec_client import *
sys.path.append('../fanuc_toolbox')
from fanuc_utils import *
sys.path.append('../../../ILC')
from ilc_toolbox import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *
from blending import *

data_type='curve_1'
curve_data_dir='../../../data/'+data_type+'/'
relative_path = read_csv(curve_data_dir+"/Curve_dense.csv", header=None).values

H_200id=np.loadtxt('base.csv',delimiter=',')
base2_R=H_200id[:3,:3]
base2_p=H_200id[:-1,-1]
# print(base2_R)
print(base2_p)
k,theta=R2rot(base2_R)
print(k,np.degrees(theta))

## robot
toolbox_path = '../../../toolbox/'
# robot1 = robot_obj('FANUC_m10ia',toolbox_path+'robot_info/fanuc_m10ia_robot_default_config.yml',tool_file_path=toolbox_path+'tool_info/paintgun.csv',d=50,acc_dict_path=toolbox_path+'robot_info/m10ia_acc_compensate.pickle',j_compensation=[1,1,-1,-1,-1,-1])
robot_flange=Transform(Ry(np.pi/2)@Rz(np.pi),[0,0,0])
robot_tcp=Transform(wpr2R([-89.895,84.408,-67.096]),[316.834,0.39,5.897])
robot_flange_tcp=robot_flange*robot_tcp
robot1=m10ia(R_tool=robot_flange_tcp.R,p_tool=robot_flange_tcp.p,d=0,acc_dict_path=toolbox_path+'robot_info/m10ia_acc.pickle')
robot2=robot_obj('FANUC_lrmate200id',toolbox_path+'robot_info/fanuc_lrmate200id_robot_default_config.yml',tool_file_path='tcp_workpiece.csv',base_transformation_file='base.csv',acc_dict_path=toolbox_path+'robot_info/lrmate200id_acc_compensate.pickle',j_compensation=[1,1,-1,-1,-1,-1])

def get_path(filename):
    df = read_csv(filename, sep =",")

    q1_1=df['J11'].tolist()
    q1_2=df['J12'].tolist()
    q1_3=df['J13'].tolist()
    q1_4=df['J14'].tolist()
    q1_5=df['J15'].tolist()
    q1_6=df['J16'].tolist()
    q2_1=df['J21'].tolist()
    q2_2=df['J22'].tolist()
    q2_3=df['J23'].tolist()
    q2_4=df['J24'].tolist()
    q2_5=df['J25'].tolist()
    q2_6=df['J26'].tolist()
    timestamp=np.round(np.array(df['timestamp'].tolist()).astype(float)*1e-3,3) # from msec to sec

    curve_exe_js1=np.radians(np.vstack((q1_1,q1_2,q1_3,q1_4,q1_5,q1_6)).T.astype(float))
    curve_exe_js2=np.radians(np.vstack((q2_1,q2_2,q2_3,q2_4,q2_5,q2_6)).T.astype(float))

    relative_path_exe=[]
    relative_path_exe_R=[]
    act_speed=[0]
    lam=[0]
    timestamp_act=[]
    last_cont = False
    for i in range(len(curve_exe_js1)):
        if i>2 and i<len(curve_exe_js1)-2:
            # if the recording is not fast enough
            # then having to same logged joint angle
            # do interpolation for estimation
            if np.all(curve_exe_js1[i]==curve_exe_js1[i+1]) and np.all(curve_exe_js2[i]==curve_exe_js2[i+1]):
                last_cont = True
                continue
        pose1_now=robot1.fwd(curve_exe_js1[i])
        timestamp_act.append(timestamp[i])
        # curve in robot's own frame
        # curve_exe1.append(pose1_now.p)
        # curve_exe2.append(pose2_now.p)
        # curve_exe_R1.append(pose1_now.R)
        # curve_exe_R2.append(pose2_now.R)

        pose2_world_now=robot2.fwd(curve_exe_js2[i],world=True)
        relative_path_exe.append(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p))
        relative_path_exe_R.append(pose2_world_now.R.T@pose1_now.R)

        if i>0:
            lam.append(lam[-1]+np.linalg.norm(relative_path_exe[-1]-relative_path_exe[-2]))
            try:
                if timestamp[-1]!=timestamp[-2]:
                    if last_cont:
                        timestep=timestamp[i]-timestamp[i-2]
                    else:
                        timestep=timestamp[i]-timestamp[i-1]
                    # print(timestep)
                    act_speed.append(np.linalg.norm(relative_path_exe[-1]-relative_path_exe[-2])/timestep)
            except IndexError:
                pass
        last_cont = False
    
    return np.array(relative_path_exe),np.array(relative_path_exe_R),np.array(lam),np.array(act_speed),np.array(timestamp_act)

def chop_extension_dual(lam, speed, timestamp, relative_path_exe,relative_path_exe_R,p_start,p_end):
        start_idx=np.argmin(np.linalg.norm(p_start-relative_path_exe,axis=1))+1
        # start_idx=0
        end_idx=np.argmin(np.linalg.norm(p_end-relative_path_exe,axis=1))
        # end_idx=-1

        #make sure extension doesn't introduce error
        if np.linalg.norm(relative_path_exe[start_idx]-p_start)>0.5:
            start_idx+=1
        if np.linalg.norm(relative_path_exe[end_idx]-p_end)>0.5:
            end_idx-=1

        timestamp=timestamp[start_idx:end_idx+1]
        relative_path_exe=relative_path_exe[start_idx:end_idx+1]
        relative_path_exe_R=relative_path_exe_R[start_idx:end_idx+1]
        speed=speed[start_idx:end_idx+1]
        lam=calc_lam_cs(relative_path_exe)

        return lam, speed, timestamp, relative_path_exe,relative_path_exe_R


relative_path_exe=[]
relative_path_exe_R=[]
lam=[]
speed=[]
timestamps=[]
# for i in range(1,9):
#     relative_path_exe_1,relative_path_exe_R_1,lam_1,speed_1,timestamp_1=get_path('LOG_100_'+str(i)+'.TXT')
#     relative_path_exe.append(relative_path_exe_1)
#     relative_path_exe_R.append(relative_path_exe_R_1)
#     lam.append(lam_1)
#     speed.append(speed_1)
#     timestamps.append(timestamp_1)
relative_path_exe_1,relative_path_exe_R_1,lam_1,speed_1,timestamp_1=get_path('LOG_300_sim.TXT')
relative_path_exe.append(relative_path_exe_1)
relative_path_exe_R.append(relative_path_exe_R_1)
lam.append(lam_1)
speed.append(speed_1)
timestamps.append(timestamp_1)

# relative_path_exe_2,relative_path_exe_R_2,lam_2,speed_2,timestamp_2=get_path('LOG_200_sim.TXT')

ax = plt.axes(projection='3d')
ax.plot3D(relative_path[:,0], relative_path[:,1],relative_path[:,2], 'red')
for i in range(1):
    ax.plot3D(relative_path_exe[i][:,0], relative_path_exe[i][:,1],relative_path_exe[i][:,2], label=str(i))
plt.legend()
plt.show()

use_id=0
lam, speed, timestamp, relative_path_exe,relative_path_exe_R=chop_extension_dual(lam[use_id], speed[use_id], timestamps[use_id], relative_path_exe[use_id],relative_path_exe_R[use_id],relative_path[0,:3],relative_path[-1,:3])
ave_speed=np.mean(speed)
error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])
print('Max Error:',max(error),'Ave. Speed:',ave_speed,'Std. Speed:',np.std(speed),'Std/Ave (%):',np.std(speed)/ave_speed*100)
print('Max Speed:',max(speed),'Min Speed:',np.min(speed),'Ave. Error:',np.mean(error),'Min Error:',np.min(error),"Std. Error:",np.std(error))
print('Max Ang Error:',max(np.degrees(angle_error)),'Min Ang Error:',np.min(np.degrees(angle_error)),'Ave. Ang Error:',np.mean(np.degrees(angle_error)),"Std. Ang Error:",np.std(np.degrees(angle_error)))
print("===========================================")

##############################plot error#####################################
fig, ax1 = plt.subplots(figsize=(6,4))
ax2 = ax1.twinx()
ax1.plot(lam, speed, 'g-', label='Speed')
ax2.plot(lam, error, 'b-',label='Error')
ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
draw_speed_max=max(speed)*1.05
ax1.axis(ymin=0,ymax=draw_speed_max)
draw_error_max=max(np.append(error,np.degrees(angle_error)))*1.05
ax2.axis(ymin=0,ymax=draw_error_max)

ax1.set_xlabel('lambda (mm)')
ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
plt.title("Speed and Error Plot")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc=1)

plt.show()