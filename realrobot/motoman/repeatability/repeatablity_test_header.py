import numpy as np
from general_robotics_toolbox import *
import sys
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from realrobot import *
from MotionSend_motoman import *
from lambda_calc import *

def main():
    robot=robot_obj('MA2010_A0',def_path='../../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../../config/weldgun2.csv',\
    pulse2deg_file_path='../../../config/MA2010_A0_pulse2deg.csv',d=50)
    ms = MotionSend(robot)

    dataset='curve_1/'
    solution_dir='baseline_motoman/'
    data_dir='../../../data/'+dataset+solution_dir
    cmd_dir=data_dir+'100L/'

    curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
    breakpoints,primitives, p_bp,q_bp=ms.extract_data_from_cmd(cmd_dir+"command.csv")
    p_bp,q_bp,primitives,breakpoints=ms.extend2(robot,q_bp,primitives,breakpoints,p_bp)

   
    error_threshold=0.5
    angle_threshold=np.radians(3)

    # speed=[100,150,200,250,300]
    # num_runs=[4,5,6]
    speed=[200]
    num_runs=[2]
    z=8
    for N in range(2):
        for v in speed:  
            ms=MotionSend(robot)
            log_results=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,v,z)
            ##############################data analysis#####################################
            lam, curve_exe, curve_exe_R,curve_exe_js, exe_speed, timestamp=ms.logged_data_analysis(robot,log_results,realrobot=True)
            #############################chop extension off##################################
            lam, curve_exe, curve_exe_R,curve_exe_js, exe_speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, exe_speed, timestamp,curve[0,:3],curve[-1,:3])
            error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
            ######################################PLOT############################
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(lam, exe_speed, 'g-', label='Speed')
            ax2.plot(lam, error, 'b-',label='Error')
            ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
            ax2.axis(ymin=0,ymax=5)
            # ax1.axis(ymin=0,ymax=1.2*v)

            ax1.set_xlabel('lambda (mm)')
            ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
            ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
            plt.title("Speed and Error Plot")
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1+h2, l1+l2, loc=1)

            plt.show()
            # plt.savefig(recorded_dir+'run_'+str(i))
            # plt.clf()
if __name__ == '__main__':
    main()