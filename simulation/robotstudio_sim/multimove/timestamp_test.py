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

from robots_def import *
from error_check import *
from MotionSend import *
from dual_arm import *

def main():
    for k in range(10):
        dataset='from_NX/'
        data_dir="../../../data/"+dataset
        solution_dir=data_dir+'dual_arm/'+'diffevo_pose2_2/'
        cmd_dir=solution_dir+'30L/'
        
        robot1=robot_obj('../../../config/abb_6640_180_255_robot_default_config.yml',tool_file_path='../../../config/paintgun.csv',d=50,acc_dict_path='')
        robot2=robot_obj('../../../config/abb_1200_5_90_robot_default_config.yml',tool_file_path=solution_dir+'tcp.csv',base_transformation_file=solution_dir+'base.csv',acc_dict_path='')


        relative_path,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,data_dir,solution_dir,robot1,robot2)

        ms = MotionSend()


        breakpoints1,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(cmd_dir+'command1.csv')
        breakpoints2,primitives2,p_bp2,q_bp2=ms.extract_data_from_cmd(cmd_dir+'command2.csv')

        breakpoints1[1:]=breakpoints1[1:]-1
        breakpoints2[2:]=breakpoints2[2:]-1

        ###get lambda at each breakpoint
        lam_bp=lam_relative_path[np.append(breakpoints1[0],breakpoints1[1:]-1)]

        vd_relative=500

        s1_all,s2_all=calc_individual_speed(vd_relative,lam1,lam2,lam_relative_path,breakpoints1)
        v2_all=[]
        v1=vmax
        for i in range(len(breakpoints1)):
            v2_all.append(speeddata(s2_all[i],9999999,9999999,999999))
            # v2_all.append(v5000)

        s1_cmd,s2_cmd=cmd_speed_profile(breakpoints1,s1_all,s2_all)

        zone=50
        z= zonedata(False,zone,1.5*zone,1.5*zone,0.15*zone,1.5*zone,0.15*zone)

        z1_all=[z]*len(v2_all)
        z2_all=[z]*len(v2_all)

        z1_all[3:7]=[z5]*4
        z2_all[3:7]=[z5]*4

        ###extension
        p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(robot1,p_bp1,q_bp1,primitives1,robot2,p_bp2,q_bp2,primitives2,breakpoints1)

        log_results=ms.exec_motions_multimove(robot1,robot2,primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,v1,v2_all,z1_all,z2_all)

        # plt.plot(np.diff(log_results.data[:,0]))
        plt.plot(log_results.data[:,0],label='traj'+str(k))

    plt.legend()
    plt.show()

        
if __name__ == "__main__":
    main()