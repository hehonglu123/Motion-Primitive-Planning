import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from scipy.signal import find_peaks
import os
cwd = os.getcwd()

sys.path.append('../../toolbox')
from robots_def import *
from error_check import *
from MotionSend_motoman import *
from MocapPoseListener import *
from lambda_calc import *
from blending import *
from realrobot import *
from PH_interp import *

def improve_perc(origin,final):
    
    return round((origin-final)/origin*100)

def main():
    curve_name='curve_2'
    dataset=curve_name+'/'
    solution_dir='baseline_motoman/'
    curve_dir="../../data/"+dataset+solution_dir
    
    # data_dir=curve_name+'_baseline_mocap_5run/'
    data_dir='curve2_baseline_mocap_5run/'
    
    config_dir='../../config/'
    robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=50,  base_marker_config_file=config_dir+'MA2010_marker_config.yaml',\
		tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

    curve = read_csv(curve_dir+"Curve_in_base_frame.csv",header=None).values

    multi_peak_threshold=0.4
    
    z=None

    gamma_v_max=1
    gamma_v_min=0.2
    
    ms = MotionSend(robot)

    ###TODO: extension fix start point, moveC support
    iteration=5

    N=5 	###N-run average

    for i in range(iteration):
        print("===== Iteration "+str(i)+"=============")
        breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd(cwd+'/'+data_dir+'command'+str(i)+'.csv')

        curve_exe, curve_exe_R, timestamp=\
            average_N_exe_mocap_read(ms,p_bp,curve,data_path=data_dir+'iteration_'+str(i)+'/',N=N)
        lam=calc_lam_cs(curve_exe)
        speed=get_speed(curve_exe,timestamp)
        
        ##############################calcualte error########################################
        error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
        
        print("Max error:",max(error))
        print("Mean error:",np.mean(error))
        print("Max ang error:",max(angle_error))
        print("Mean ang error:",np.mean(angle_error))
        
        if i==0:
            maxerr_init=max(error)
            meanerr_init=np.mean(error)
            maxangerr_init=np.degrees(max(angle_error))
            meanangerr_init=np.degrees(np.mean(angle_error))
    
    print("")
    print(data_dir)
    print("Max error improvement:",improve_perc(maxerr_init,max(error)),"%")
    print("Mean error improvement:",improve_perc(meanerr_init,np.mean(error)),"%")
    print("Max Angular error improvement:",improve_perc(maxangerr_init,np.degrees(max(angle_error))),"%")
    print("Mean error improvement:",improve_perc(meanangerr_init,np.degrees(np.mean(angle_error))),"%")

if __name__ == "__main__":
    main()