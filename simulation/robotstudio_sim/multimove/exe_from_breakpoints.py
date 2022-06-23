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

# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *

def main():
    dataset='wood'
    relative_path = read_csv("../../../data/"+dataset+"/relative_path_tool_frame.csv", header=None).values

    #kin def
    robot1=abb1200(d=50)
    robot2=abb6640()
    base2_R=np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    base2_p=np.array([3000,1000,0])

    ms = MotionSend(robot1=robot1,robot2=robot2,tool1=tooldata(True,pose([75,0,493.30127019],[quatR[0],quatR[1],quatR[2],quatR[3]]),loaddata(1,[0,0,0.001],[1,0,0,0],0,0,0)),tool2=tooldata(True,pose([50,0,450],[quatR[0],quatR[1],quatR[2],quatR[3]]),loaddata(1,[0,0,0.001],[1,0,0,0],0,0,0)))
    data_dir="../../../data/wood/dual_arm/"

    v = speeddata(250,9999999,9999999,999999)
    speed={'v1000':v1000}
    # speed={'v50':v50,'v100':v100,'v200':v200,'v400':v400,'v500':v500,'vmax':vmax}
    zone={'z10':z10}

    for s in speed:
        for z in zone: 
            breakpoints1,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(data_dir+'command1.csv')
            breakpoints2,primitives2,p_bp2,q_bp2=ms.extract_data_from_cmd(data_dir+'command2.csv')
            logged_data=ms.exec_motions_multimove(breakpoints1,primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,v,v,z10,z10)
            StringData=StringIO(logged_data)
            df = read_csv(StringData, sep =",")
            lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, act_speed, timestamp, relative_path_exe,relative_path_exe_R = ms.logged_data_analysis_multimove(df,base2_R,base2_p)

            ###calculate error
            error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])

            fig, ax1 = plt.subplots()

            ax2 = ax1.twinx()
            ax1.plot(lam[1:],act_speed, 'g-', label='Speed')
            ax2.plot(lam, error, 'b-',label='Error')
            ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')

            ax1.set_xlabel('lambda (mm)')
            ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
            ax2.set_ylabel('Error (mm)', color='b')
            plt.title("Speed: "+data_dir+s+'_'+z)
            ax1.legend(loc=0)

            ax2.legend(loc=0)

            plt.legend()
            plt.show()
if __name__ == "__main__":
    main()