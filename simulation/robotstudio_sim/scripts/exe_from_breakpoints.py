########
# This module utilized https://github.com/johnwason/abb_motion_program_exec
# and send whatever the motion primitives that algorithms generate
# to RobotStudio
########

import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from io import StringIO
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *

def main():
    ms = MotionSend()
    # ms = MotionSend(url='http://192.168.55.1:80')
    dataset='curve_1/'
    solution_dir='curve_pose_opt1/'

    data_dir='../../../data/'+dataset+solution_dir
    cmd_dir=data_dir+'100L/'

    curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
    lam_original=calc_lam_cs(curve[:,:3])

    robot=robot_obj('ABB_6640_180_255','../../../config/abb_6640_180_255_robot_default_config.yml',tool_file_path='../../../config/paintgun.csv',d=50,acc_dict_path='')

    v250 = speeddata(250,9999999,9999999,999999)
    v350 = speeddata(350,9999999,9999999,999999)
    v450 = speeddata(450,9999999,9999999,999999)
    v900 = speeddata(900,9999999,9999999,999999)
    v1100 = speeddata(1100,9999999,9999999,999999)
    v1200 = speeddata(1200,9999999,9999999,999999)
    v1300 = speeddata(1300,9999999,9999999,999999)
    v1400 = speeddata(1400,9999999,9999999,999999)
    # speed={'v200':v200,'v250':v250,'v300':v300,'v350':v350,'v400':v400,'v450':v450,'v500':v500}
    # speed={'v800':v800,'v900':v900,'v1000':v1000,'v1100':v1100,'v1200':v1200,'v1300':v1300,'v1400':v1400}
    # speed={'v800':v800,'v1000':v1000,'v1200':v1200,'v1500':v1500,'v2000':v2000}
    speed={'v400':v400}
    zone={'z100':z100}

    for s in speed:
        for z in zone: 
            breakpoints,primitives, p_bp,q_bp=ms.extract_data_from_cmd(cmd_dir+"command.csv")
            q_bp_end=q_bp[-1][0]
            p_bp, q_bp = ms.extend(robot, q_bp, primitives, breakpoints, p_bp,extension_start=100,extension_end=100)
            log_results = ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,speed[s],zone[z])

            ##############################data analysis#####################################
            lam, curve_exe, curve_exe_R,curve_exe_js, exe_speed, timestamp=ms.logged_data_analysis(robot,log_results,realrobot=True)
            #############################chop extension off##################################
            lam, curve_exe, curve_exe_R,curve_exe_js, exe_speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, exe_speed, timestamp,curve[0,:3],curve[-1,:3])
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

            ####joint limit visualization
            # for i in range(len(curve_exe_js[0])):
            #     plt.plot(curve_exe_js[:,i])
            #     plt.title('joint'+str(i+1))
            #     plt.ylim([robot.lower_limit[i],robot.upper_limit[i]])
            #     plt.show()



            #############SAVE DATA##############################
            # f = open(data_dir+"curve_exe"+"_"+s+"_"+z+".csv", "w")
            # f.write(logged_data)
            # f.close()

if __name__ == "__main__":
    main()