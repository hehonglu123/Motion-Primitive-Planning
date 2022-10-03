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
    dataset='from_NX/'
    data_dir="../../../data/"+dataset
    solution_dir=data_dir+'dual_arm/'+'diffevo_pose2/'
    cmd_dir=solution_dir+'30L/'
    
    relative_path,robot1,robot2,base2_R,base2_p,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,data_dir,solution_dir)

    ms = MotionSend(robot1=robot1,robot2=robot2,base2_R=base2_R,base2_p=base2_p)


   

    breakpoints1,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(cmd_dir+'command1.csv')
    breakpoints2,primitives2,p_bp2,q_bp2=ms.extract_data_from_cmd(cmd_dir+'command2.csv')

    breakpoints1[1:]=breakpoints1[1:]-1
    breakpoints2[2:]=breakpoints2[2:]-1

    ###get lambda at each breakpoint
    lam_bp=lam_relative_path[np.append(breakpoints1[0],breakpoints1[1:]-1)]

    vd_relative=2500

    s1_all,s2_all=calc_individual_speed(vd_relative,lam1,lam2,lam_relative_path,breakpoints1)
    v2_all=[]
    v1=vmax
    for i in range(len(breakpoints1)):
        v2_all.append(speeddata(s2_all[i],9999999,9999999,999999))
        # v2_all.append(v5000)

    s1_cmd,s2_cmd=cmd_speed_profile(breakpoints1,s1_all,s2_all)

    zone=50
    z= zonedata(False,zone,1.5*zone,1.5*zone,0.15*zone,1.5*zone,0.15*zone)



    ###extension
    p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1)

    logged_data=ms.exec_motions_multimove(breakpoints1,primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,v1,v2_all,z,z)
    # Write log csv to file
    with open("recorded_data/curve_exe_v"+str(vd_relative)+'_z'+str(zone)+'.csv',"w") as f:
        f.write(logged_data)
    StringData=StringIO(logged_data)
    df = read_csv(StringData, sep =",")
    lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R = ms.logged_data_analysis_multimove(df,base2_R,base2_p,realrobot=True)
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
    ax1.plot(lam_relative_path[2:],s1_cmd,'p-',label='TCP1 cmd Speed')
    ax1.plot(lam,speed2, 'm-', label='TCP2 Speed')
    ax1.plot(lam_relative_path[2:],s2_cmd,'p-',label='TCP2 cmd Speed')
    ax2.plot(lam, error, 'b-',label='Error')
    ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')

    ax1.set_xlabel('lambda (mm)')
    ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
    ax2.set_ylabel('Error (mm)', color='b')
    plt.title("Speed: "+dataset+'v'+str(vd_relative)+'_z'+str(zone))
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc=1)


    plt.savefig('recorded_data/curve_exe_v'+str(vd_relative)+'_z'+str(zone))
    plt.show()
if __name__ == "__main__":
    main()