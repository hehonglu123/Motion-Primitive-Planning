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

robot=abb6640(d=50)

dataset='wood/'
dataset_dir="../data/"+dataset
curve = read_csv(dataset_dir+"Curve_in_base_frame.csv",header=None).values

data_dir="recorded_data/repeatibility_EGM/"+dataset

speed=[100,200,300,400]




for v in speed:
    trajectories=[]

    trajectory_js0 = read_csv(dataset_dir+'_v'+str(v)+'_iteration0.csv',header=None).values
    curve_exe_it0=[]
    for q in trajectory_js0:
        curve_exe_it0.append(robot.fwd(q).p)

    for i in range(1,5):
        trajectory_js = read_csv(dataset_dir+'_v'+str(v)+'_iteration'+str(i)+'.csv',header=None).values
        trajectory=[]
        error=[]
        for q in trajectory_js:
            trajectory.append(robot.fwd(q).p)
            error.append(calc_error_backup(trajectory[-1],curve_exe_it0))
        lam=calc_lam_cs(np.array(trajectory))
        plt.plot(lam,error,label=str(i)+'th trajectory')
    
    plt.xlabel('Lambda (mm)')
    plt.ylabel('Error (mm)', color='g')

    plt.title("Repeatibility: "+dataset+'_'+'v'+str(s))

    plt.legend()
    plt.show()


