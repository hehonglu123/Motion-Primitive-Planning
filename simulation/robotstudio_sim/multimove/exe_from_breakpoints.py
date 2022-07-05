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
    dataset='wood/'
    data_dir="../../../data/"+dataset
    relative_path = read_csv(data_dir+"/relative_path_tool_frame.csv", header=None).values

    with open(data_dir+'dual_arm/abb6640.yaml') as file:
        H_6640 = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

    base2_R=H_6640[:3,:3]
    base2_p=1000*H_6640[:-1,-1]
    ms = MotionSend()

    s=1000
    z=10
    v = speeddata(s,9999999,9999999,999999)


    breakpoints1,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(data_dir+'/dual_arm/command1.csv')
    breakpoints2,primitives2,p_bp2,q_bp2=ms.extract_data_from_cmd(data_dir+'/dual_arm/command2.csv')

    logged_data=ms.exec_motions_multimove(breakpoints1,primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,v,v,z10,z10)

    StringData=StringIO(logged_data)
    df = read_csv(StringData, sep =",")
    lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, act_speed, timestamp, relative_path_exe,relative_path_exe_R = ms.logged_data_analysis_multimove(df,base2_R,base2_p)
    print(np.degrees(curve_exe_js2))
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
    plt.title("Speed: "+dataset+str(s)+'_z'+str(z))
    ax1.legend(loc=0)

    ax2.legend(loc=0)

    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()