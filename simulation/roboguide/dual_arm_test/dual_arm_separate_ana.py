
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

all_data_type=['notsparate','connect','thread']
all_joints_iter={}
all_stamp_iter={}
all_iteraion=4
for data_type in all_data_type:
    data_dir = 'data/separate_controller_'+data_type+'/'

    joints_iter=[]
    stamps_iter=[]
    fig1, ax1 = plt.subplots(6,1)
    fig2, ax2 = plt.subplots(6,1)
    for i in range(all_iteraion):
        data = np.array(read_csv(data_dir+"/iter_"+str(i)+"_curve_exe_js.csv", header=None).values)
        stamps_iter.append(data[:,0])
        joints_iter.append(data[:,1:])

        for j in range(6):
            ax1[j].scatter(data[:,0],data[:,j+1],s=1,label='Trial '+str(i+1))
            ax2[j].scatter(data[:,0],data[:,j+1+6],s=1,label='Trial '+str(i+1))
    fig1.suptitle("Arm 1, Joint 1~6 Trajectory Across Trials, "+data_type)
    fig2.suptitle("Arm 2, Joint 1~6 Trajectory Across Trials, "+data_type)
    plt.legend()
    plt.show()

    all_joints_iter[data_type]=joints_iter
    all_stamp_iter[data_type]=stamps_iter

fig1, ax1 = plt.subplots(6,1)
fig2, ax2 = plt.subplots(6,1)
for data_type in all_data_type:
    for j in range(6):
        ax1[j].scatter(all_stamp_iter[data_type][0],all_joints_iter[data_type][0][:,j],s=1,label='Type '+data_type)
        ax2[j].scatter(all_stamp_iter[data_type][0],all_joints_iter[data_type][0][:,j+6],s=1,label='Type '+data_type)
fig1.suptitle("Arm 1, Joint 1~6 Trajectory Across Trials, "+data_type)
fig2.suptitle("Arm 2, Joint 1~6 Trajectory Across Trials, "+data_type)
plt.legend()
plt.show()