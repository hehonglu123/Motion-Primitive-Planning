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

data_dir="recorded_data/repeatibility/"+dataset

# speed=[200,400,600,800]
speed=[50,100,150,200]

zone=['z10']
ms = MotionSend()


for s in speed:
    for z in zone:
        ###use iteration 0 as baseline

        ###read in curve_exe
        df = read_csv(data_dir+"v"+str(s)+'_iteration0.csv')
        ##############################data analysis#####################################
        lam_it0, curve_exe_it0, curve_exe_R_it0,curve_exe_js_it0, speed_it0, timestamp_it0=ms.logged_data_analysis(robot,df,realrobot=True)
        #############################chop extension off##################################
        lam_it0, curve_exe_it0, curve_exe_R_it0,curve_exe_js_it0, speed_it0, timestamp_it0=ms.chop_extension(curve_exe_it0, curve_exe_R_it0,curve_exe_js_it0, speed_it0, timestamp_it0,curve[0,:3],curve[-1,:3])



        curve_diff=[]
        plt.figure()
        for i in range(1,5):
            ##read in curve_exe
            df = read_csv(data_dir+"v"+str(s)+'_iteration'+str(i)+'.csv')
            ##############################data analysis#####################################
            lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)

            #############################chop extension off##################################
            lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])

            error=[]
            for exe_p in curve_exe:
                error.append(calc_error_backup(exe_p,curve_exe_it0))

            
            plt.plot(lam,error,label=str(i)+'th trajectory')
        plt.xlabel('Lambda (mm)')
        plt.ylabel('Error (mm)', color='g')

        plt.title("Repeatibility: "+dataset+'_'+'v'+str(s))

        plt.legend()
        plt.show()


