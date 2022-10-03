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
    ms = MotionSend()
    dataset='from_NX/'
    data_dir="../../../data/"+dataset
    solution_dir=data_dir+'dual_arm/'+'diffevo3/'
    cmd_dir=solution_dir+'30L/'

    relative_path,robot1,robot2,base2_R,base2_p,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,data_dir,solution_dir)

    curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
    lam_original=calc_lam_cs(curve[:,:3])

    robot=robot2

    v=v1000
    z=z10

    breakpoints,primitives, p_bp,q_bp=ms.extract_data_from_cmd(cmd_dir+"command2.csv")
    p_bp, q_bp = ms.extend(robot, q_bp, primitives, breakpoints, p_bp,extension_start=100,extension_end=100)

    logged_data= ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,v,z)

    StringData=StringIO(logged_data)
    df = read_csv(StringData, sep =",")
    ##############################data analysis#####################################
    lam, curve_exe, curve_exe_R,curve_exe_js, exe_speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)
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
    for bp in breakpoints:
        plt.axvline(x=lam_original[bp])

    plt.legend()
    plt.show()

    ####joint limit visualization
    for i in range(len(curve_exe_js[0])):
        plt.plot(curve_exe_js[:,i])
        plt.title('joint'+str(i+1))
        plt.ylim([robot.lower_limit[i],robot.upper_limit[i]])
        plt.show()



    #############SAVE DATA##############################
    # f = open(data_dir+"curve_exe"+"_"+s+"_"+z+".csv", "w")
    # f.write(logged_data)
    # f.close()

if __name__ == "__main__":
    main()