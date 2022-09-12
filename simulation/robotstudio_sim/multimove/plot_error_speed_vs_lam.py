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


from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *
from dual_arm import *

def main():
    dataset='from_NX/'
    data_dir="../../../data/"+dataset
    solution_dir=data_dir+'dual_arm/'+'diffevo3/'
    cmd_dir=solution_dir+'greedy0.2/'
    
    relative_path,robot1,robot2,base2_R,base2_p,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,data_dir,solution_dir,cmd_dir)

    ms = MotionSend(robot1=robot1,robot2=robot2,base2_R=base2_R,base2_p=base2_p)


    ###read in curve_exe
    df = read_csv('recorded_data/curve_exe_v5000_z10.csv')

    lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R =ms.logged_data_analysis_multimove(df,base2_R,base2_p,realrobot=True)
    #############################chop extension off##################################
    lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
        ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])
    
    speed1=get_speed(curve_exe1,timestamp)
    speed2=get_speed(curve_exe2,timestamp)
    ###calculate error
    error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(lam,speed, 'g-', label='Relative Speed')
    ax1.plot(lam,speed1, 'r-', label='TCP1 Speed')
    ax1.plot(lam,speed2, 'm-', label='TCP2 Speed')
    ax2.plot(lam, error, 'b-',label='Error')
    ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')

    ax1.set_xlabel('lambda (mm)')
    ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
    ax2.set_ylabel('Error (mm)', color='b')
    plt.title('Exe Profile')
    ax1.legend(loc=0)

    ax2.legend(loc=0)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()