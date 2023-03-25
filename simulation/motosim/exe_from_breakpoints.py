import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys

sys.path.append('../../toolbox')
from robots_def import *
from error_check import *
from MotionSend_motoman import *

def main():
    robot=robot_obj('MA2010_A0',def_path='../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun2.csv',\
    pulse2deg_file_path='../../config/MA2010_A0_pulse2deg.csv',d=50)

    ms = MotionSend(robot)

    dataset='curve_2/'
    solution_dir='baseline_motoman/'

    data_dir='../../data/'+dataset+solution_dir
    cmd_dir=data_dir+'50L/'

    curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
    lam_original=calc_lam_cs(curve[:,:3])

    

    speed={'v50':50}

    for s in speed:
        breakpoints,primitives, p_bp,q_bp=ms.extract_data_from_cmd(cmd_dir+"command.csv")
        q_bp_end=q_bp[-1][0]
        # p_bp, q_bp = ms.extend(robot, q_bp, primitives, breakpoints, p_bp,extension_start=150,extension_end=100)
        log_results = ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,[5]+[speed[s]]*len(breakpoints))

        ##############################data analysis#####################################
        lam, curve_exe, curve_exe_R,curve_exe_js, exe_speed, timestamp=ms.logged_data_analysis(robot,log_results,realrobot=True)
        #############################chop extension off##################################
        # lam, curve_exe, curve_exe_R,curve_exe_js, exe_speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, exe_speed, timestamp,curve[0,:3],curve[-1,:3])
        ##############################calcualte error########################################
        error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(lam, exe_speed, 'g-', label='Speed')
        ax2.plot(lam, error, 'b-',label='Error')
        ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
        ax2.axis(ymin=0,ymax=6)
        ax1.axis(ymin=0,ymax=1500)

        ax1.set_xlabel('lambda (mm)')
        ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
        ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
        plt.title("Speed and Error Plot")
        ax1.legend(loc="upper right")

        ax2.legend(loc="upper left")

        ###plot breakpoints index
        breakpoints[1:]=breakpoints[1:]-1
        for bp in breakpoints:
            plt.axvline(x=lam_original[bp])

        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()