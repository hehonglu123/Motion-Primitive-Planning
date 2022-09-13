from cProfile import label
from copy import deepcopy
import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv,DataFrame
import sys
from io import StringIO
from scipy.signal import find_peaks
import yaml
from matplotlib import pyplot as plt
from pathlib import Path
from fanuc_motion_program_exec_client import *
sys.path.append('../fanuc_toolbox')
from fanuc_utils import *
sys.path.append('../../../ILC')
from ilc_toolbox import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *
from blending import *

data_type='blade_base_shift'

# data and curve directory
if data_type=='blade':
    curve_data_dir='../../../data/from_NX/'
    cmd_dir='../data/curve_blade/'
    data_dir='data/blade_dual/'
elif data_type=='wood':
    curve_data_dir='../../../data/wood/'
    cmd_dir='../data/curve_wood/'
    data_dir='data/wood_dual/'
elif data_type=='blade_arm_shift':
    curve_data_dir='../../../data/from_NX/'
    cmd_dir='../data/curve_blade_arm_shift/'
    data_dir='data/blade_arm_shift_dual/'
elif data_type=='blade_base_shift':
    curve_data_dir='../../../data/from_NX/'
    cmd_dir='../data/curve_blade_base_shift/'
    data_dir='data/blade_base_shift_dual/'

test_type='dual_arm'

cmd_dir=cmd_dir+test_type+'/'
# the result data folder
data_dir=data_dir+'results_1000_dual_arm_extend1_nospeedreg/'

# relative path
relative_path = np.array(read_csv(curve_data_dir+"/Curve_dense.csv", header=None).values)

# the second robot relative to the fist robot
with open(cmd_dir+'../m900ia.yaml') as file:
    H_robot2 = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
base2_R=H_robot2[:3,:3]
base2_p=1000*H_robot2[:-1,-1]
base2_T=rox.Transform(base2_R,base2_p)
# workpiece (curve) relative to robot tcp
with open(cmd_dir+'../tcp.yaml') as file:
    H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

# define robot
robot1=m710ic(d=50)
robot2=m900ia(R_tool=H_tcp[:3,:3],p_tool=H_tcp[:-1,-1])

# fanuc motion send tool
if data_type=='blade':
    ms = MotionSendFANUC(robot1=robot1,robot2=robot2)
elif data_type=='wood':
    ms = MotionSendFANUC(robot1=robot1,robot2=robot2,utool2=3)
elif data_type=='blade_arm_shift':
    ms = MotionSendFANUC(robot1=robot1,robot2=robot2,utool2=6)
elif data_type=='blade_base_shift':
    ms = MotionSendFANUC(robot1=robot1,robot2=robot2,utool2=2)

## read data
all_curve_exe_js1 = [np.array(read_csv(data_dir+"iter_27_curve_exe_js1.csv", header=None).values),\
                    np.array(read_csv(data_dir+"iter_28_curve_exe_js1.csv", header=None).values)]
all_curve_exe_js2 = [np.array(read_csv(data_dir+"iter_27_curve_exe_js2.csv", header=None).values),\
                    np.array(read_csv(data_dir+"iter_28_curve_exe_js2.csv", header=None).values)]
all_timestamps = [np.array(read_csv(data_dir+"iter_27_timestamp.csv", header=None).values),\
                    np.array(read_csv(data_dir+"iter_28_timestamp.csv", header=None).values)]
all_curve_exe=[]
all_curve_exe_n=[]
all_error=[]
all_angle_error=[]
all_curve_exe1=[]
all_curve_exe_R1=[]
all_curve_exe2=[]
all_curve_exe_R2=[]
for i in range(len(all_curve_exe_js1)):
    exe_js1=all_curve_exe_js1[i]
    exe_js2=all_curve_exe_js2[i]
    curve_exe_R=[]
    curve_exe=[]
    curve_exe_R1=[]
    curve_exe1=[]
    curve_exe_R2=[]
    curve_exe2=[]
    for j in range(len(exe_js1)):
        this_pose1 = robot1.fwd(exe_js1[j])
        this_pose2 = robot1.fwd(exe_js2[j])
        curve_exe1.append(np.append(this_pose1.p,np.rad2deg(R2rpy(this_pose1.R))))
        curve_exe_R1.append(R2rpy(this_pose1.R))
        curve_exe2.append(np.append(this_pose2.p,np.rad2deg(R2rpy(this_pose2.R))))
        curve_exe_R2.append(R2rpy(this_pose2.R))

        pose2_world_now=robot2.fwd(exe_js2[j],base2_R,base2_p)
        relative_p = np.dot(pose2_world_now.R.T,this_pose1.p-pose2_world_now.p)
        relative_R = pose2_world_now.R.T@this_pose1.R
        curve_exe.append(np.append(relative_p,relative_R[:,-1]))
        curve_exe_R.append(relative_R[:,-1])

    curve_exe=np.array(curve_exe)
    curve_exe_R=np.array(curve_exe_R)
    curve_exe1=np.array(curve_exe1)
    curve_exe_R1=np.array(curve_exe_R1)
    curve_exe2=np.array(curve_exe2)
    curve_exe_R2=np.array(curve_exe_R2)
    error,angle_error=calc_all_error_w_normal(curve_exe[:,:3],relative_path[:,:3],curve_exe[:,3:],relative_path[:,3:])

    all_curve_exe.append(curve_exe)
    all_curve_exe_n.append(curve_exe_R)
    all_error.append(error)
    all_angle_error.append(angle_error)
    all_curve_exe1.append(curve_exe1)
    all_curve_exe_R1.append(curve_exe_R1)
    all_curve_exe2.append(curve_exe2)
    all_curve_exe_R2.append(curve_exe_R2)

############ visualization for analyze

title_plot=['x','y','z','roll','pitch','yaw']
marker_size=5
dt=0.012
for i in range(6): # draw x y z r p y
    fig, ax = plt.subplots(3,1)
    # draw robot1
    ax[0].scatter(all_timestamps[0][-17:],all_curve_exe1[0][-17:,i],s=marker_size,label='iter 27') # iter 27
    ax[0].scatter(all_timestamps[1][-17:],all_curve_exe1[1][-17:,i],s=marker_size,label='iter 28') # iter 28
    ax[0].set_title('Robot1 '+title_plot[i])
    ax[0].legend(loc=0)
    # ax[0].axis(xmin=2.2,xmax=2.45)
    # draw robot2
    ax[1].scatter(all_timestamps[0][-17:],all_curve_exe2[0][-17:,i],s=marker_size,label='iter 27')
    ax[1].scatter(all_timestamps[1][-17:],all_curve_exe2[1][-17:,i],s=marker_size,label='iter 28')
    ax[1].set_title('Robot2 '+title_plot[i])
    ax[1].legend(loc=0)
    # ax[1].axis(xmin=2.2,xmax=2.45)
    # draw relative error
    ax[2].scatter(all_timestamps[0][-17:],all_error[0][-17:],s=marker_size,label='iter 27')
    ax[2].scatter(all_timestamps[1][-17:],all_error[1][-17:],s=marker_size,label='iter 28')
    ax[2].set_title('Relative Error '+title_plot[i])
    ax[2].legend(loc=0)
    # ax[2].axis(xmin=2.2,xmax=2.45)
    fig.suptitle('plot '+title_plot[i])
    plt.show()

    # draw differences
    fig, ax = plt.subplots(2,1)
    diff1 = []
    diff_shift1 = []
    draw_timestamp=[]
    draw_timestamp_shift=[]
    for j in range(len(all_timestamps[0])):
        stamp_id = np.argwhere(all_timestamps[1]==all_timestamps[0][j])
        if len(stamp_id)==0:
            pass
        else:
            stamp_id=stamp_id[0][0]
            diff1.append(np.fabs(all_curve_exe1[0][j,i]-all_curve_exe1[1][stamp_id,i]))
            draw_timestamp.append(all_timestamps[0][j])
        stamp_id = np.argwhere(all_timestamps[1]==(all_timestamps[0][j]+0.012))
        if len(stamp_id)==0:
            pass
        else:
            stamp_id=stamp_id[0][0]
            diff_shift1.append(np.fabs(all_curve_exe1[0][j,i]-all_curve_exe1[1][stamp_id,i]))
            draw_timestamp_shift.append(all_timestamps[0][j])
    # draw robot1 differences
    ax[0].scatter(draw_timestamp,diff1,s=marker_size,label='difference') # iter 27
    ax[0].scatter(draw_timestamp_shift,diff_shift1,s=marker_size,label='difference shift') # iter 28
    ax[0].set_title('Robot1 '+title_plot[i])
    ax[0].legend(loc=0)
    # draw relative error
    ax[1].scatter(all_timestamps[0],all_error[0],s=marker_size,label='iter 27')
    ax[1].scatter(all_timestamps[1],all_error[1],s=marker_size,label='iter 28')
    ax[1].set_title('Relative Error '+title_plot[i])
    ax[1].legend(loc=0)
    # ax[2].axis(xmin=2.2,xmax=2.45)
    fig.suptitle('plot '+title_plot[i]+' differences')
    plt.show()
