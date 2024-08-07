#move robot1 only in multimove
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
from dual_arm import *

def main():
    dataset='wood/'
    data_dir="../../../data/"+dataset
    solution_dir=data_dir+'dual_arm/'+'diffevo_pose1/'
    cmd_dir=solution_dir+'50L/'
    
    relative_path,robot1,robot2,base2_R,base2_p,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,data_dir,solution_dir,cmd_dir)

    ms = MotionSend(robot1=robot1,robot2=robot2,base2_R=base2_R,base2_p=base2_p)


   

    breakpoints1,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(cmd_dir+'command1.csv')
    breakpoints2,primitives2,p_bp2,q_bp2=ms.extract_data_from_cmd(cmd_dir+'command2.csv')

    breakpoints1[1:]=breakpoints1[1:]-1
    breakpoints2[2:]=breakpoints2[2:]-1

    

    p_bp_start=copy.deepcopy(p_bp2[0])
    p_bp_end=copy.deepcopy(p_bp2[-1])


    vd_relative=1000

    s1_all,s2_all=calc_individual_speed(vd_relative,lam1,lam2,lam_relative_path,breakpoints1)
    v1_all=[]
    v2_all=[]
    v1=vmax
    for i in range(len(breakpoints1)):
        v1_all.append(speeddata(s1_all[i],9999999,9999999,999999))
        v2_all.append(speeddata(s2_all[i],9999999,9999999,999999))
        # v2_all.append(v5000)

    s1_cmd,s2_cmd=cmd_speed_profile(breakpoints1,s1_all,s2_all)

    zone=10
    z= zonedata(False,zone,1.5*zone,1.5*zone,0.15*zone,1.5*zone,0.15*zone)



    ###extension
    p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1)

    logged_data=ms.exec_motions(robot2,breakpoints2,primitives2,p_bp2,q_bp2,v2_all,z)
    # Write log csv to file
    with open("recorded_data/curve_exe_v"+str(vd_relative)+'_z'+str(zone)+'.csv',"w") as f:
        f.write(logged_data)
    StringData=StringIO(logged_data)
    df = read_csv(StringData, sep =",")
    lam, curve_exe,curve_exe_R,curve_exe_js, speed, timestamp = ms.logged_data_analysis(robot2,df,realrobot=True)
    
    #############################chop extension off##################################

    lam, curve_exe,curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe,curve_exe_R,curve_exe_js, speed, timestamp, p_bp_start,p_bp_end)
    
    speed=get_speed(curve_exe,timestamp)
    
    fig, ax1 = plt.subplots()

    ax1.plot(lam,speed, 'g-', label='TCP')
    ax1.plot(lam2[1:],s2_cmd,'p-',label='TCP2 cmd Speed')

    ax1.set_xlabel('lambda (mm)')
    ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
    plt.title("Speed: "+dataset+'v'+str(vd_relative)+'_z'+str(zone))
    plt.legend()


    # plt.savefig('recorded_data/curve_exe_v'+str(vd_relative)+'_z'+str(zone))
    plt.show()
if __name__ == "__main__":
    main()