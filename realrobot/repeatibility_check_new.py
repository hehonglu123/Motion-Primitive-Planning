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
from utils import *



robot=abb6640(d=50)

dataset='from_NX/'
dataset_dir="../data/"+dataset
curve = read_csv(dataset_dir+"Curve_in_base_frame.csv",header=None).values

data_dir="recorded_data/repeatibility/"+dataset

speed=[200,400,600,800]
# speed=[50,100,150,200]
# speed=[150]
zone=['z10']
ms = MotionSend()


for s in speed:
    for z in zone:
        curve_exe_all=[]
        timestamp_all=[]

        # plt.figure()
        # ax = plt.axes(projection='3d')
        for i in range(5):
            ##read in curve_exe
            df = read_csv(data_dir+"v"+str(s)+'_iteration'+str(i)+'.csv')
            ##############################data analysis#####################################
            lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)
            #############################chop extension off##################################
            lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])
            timestamp=timestamp-timestamp[0]

            curve_exe_all.append(curve_exe)
            timestamp_all.append(timestamp)


            # ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c=np.random.rand(3,),label=str(i+1)+'th trajectory')

        ###infer average curve from linear interplateion
        curve_all_new, avg_curve, timestamp_d=average_curve(curve_exe_all,timestamp_all)
        # ax.plot3D(avg_curve[:,0], avg_curve[:,1], avg_curve[:,2], c=np.random.rand(3,),label='avg trajectory')

        for i in range(len(curve_all_new)):
            ###########calc error based on cmd curve#################
            # error=calc_all_error(curve_all_new[i],curve[:,:3])
            ###################calc error based on avg curve############################
            # error=calc_all_error(curve_all_new[i],avg_curve)

            error=[]
            for exe_p in curve_all_new[i]:
                error.append(calc_error_backup(exe_p,avg_curve))


            plt.plot(timestamp_d,error,label=str(i+1)+'th trajectory')


            print('curve '+str(i+1))
            print('max: ',np.max(error))
            print('avg: ',np.average(error))
            print('med: ',np.median(error))
            print('std: ',np.std(error))
            print('DONE')
            
        plt.xlabel('Time (s)')
        plt.ylabel('Error (mm)', color='g')

        plt.title("Repeatibility: "+dataset+'_'+'v'+str(s))

        plt.legend()
        plt.show()


