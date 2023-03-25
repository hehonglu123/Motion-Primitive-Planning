########
# This module utilized https://github.com/johnwason/abb_motion_program_exec
# and send whatever the motion primitives that algorithms generate
# to RobotStudio
########

import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from io import StringIO
from lambda_calc import *

def main():
    robot=abb6640(d=50)

    error_threshold=0.5
    angle_threshold=np.radians(3)

    dataset="../../../data/from_NX/"
    

    curve = read_csv(dataset+"Curve_in_base_frame.csv",header=None).values

    ms = MotionSend()
    
    # data_dir=dataset+"baseline/"+str(num_l)+'L/'
    data_dir=dataset+'robodk/'

    vmax = speeddata(10000,9999999,9999999,999999)
    v=1500
    v_prev=3000
    v_prev_possible=100
    

    max_error=999
    while True:
        ms = MotionSend()
        ###update velocity profile
        v_cmd = speeddata(v,9999999,9999999,999999)
        #execute with 10cm acc/dcc range
        breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd(data_dir+'command.csv')
        ###################ROBODK ONLY, CHOP OFF THEIR EXTENSION#######################
        primitives=primitives[1:-1]
        primitives[0]='movej_fit'
        breakpoints=breakpoints[1:-1]
        p_bp=p_bp[1:-1]
        q_bp=q_bp[1:-1]
        ###############################################################################
        ################################extension######################################
        p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp)
        ###############################################################################
        logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,v_cmd,z10)
        StringData=StringIO(logged_data)
        df = read_csv(StringData, sep =",")
        ##############################data analysis#####################################
        lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)
        lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])

        error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:],extension=True)



        start_idx=np.argmin(np.linalg.norm(curve[0,:3]-curve_exe,axis=1))
        end_idx=np.argmin(np.linalg.norm(curve[-1,:3]-curve_exe,axis=1))

        curve_exe=curve_exe[start_idx:end_idx+1]
        curve_exe_R=curve_exe_R[start_idx:end_idx+1]
        speed=speed[start_idx:end_idx+1]
        lam=calc_lam_cs(curve_exe)

        # _,speed=lfilter(timestamp[start_idx:end_idx+1],np.array(speed))
        
        max_error=max(error)
        print('cmd speed: ',v, 'max error: ',max_error, 'max ori error: ', max(angle_error), 'std(speed): ',np.std(speed))
        v_prev_temp=v
        if max_error>error_threshold or np.std(speed)>np.average(speed)/20 or max(angle_error)>angle_threshold:
            v-=abs(v_prev-v)/2
        else:
            v_prev_possible=v
            #stop condition
            if error_threshold-max_error<0.05:
                break   
            v+=abs(v_prev-v)/2

        v_prev=v_prev_temp

        #if stuck
        if abs(v-v_prev)<1:
            v=v_prev_possible
            v_cmd = speeddata(v,9999999,9999999,999999)
            logged_data=ms.exe_from_file_w_extension(data_dir+"command.csv",dataset+"Curve_js.csv",v_cmd,z10,extension_d=100)
            StringData=StringIO(logged_data)
            df = read_csv(StringData, sep =",")
            lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df)
            error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:],extension=True)
            
            max_error=max(error)
            break


        # fig, ax1 = plt.subplots()

        # ax2 = ax1.twinx()
        # ax1.plot(lam,speed, 'g-', label='Speed')
        # ax2.plot(lam, error, 'b-',label='Error')
        # ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')

        # ax1.set_xlabel('lambda (mm)')
        # ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
        # ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
        # plt.title("Speed: "+data_dir+'v_'+str(v_prev)+'_z10')
        # ax1.legend(loc=0)

        # ax2.legend(loc=0)

        # ax2.axis(ymin=0,ymax=2)
        # ax1.axis(ymin=0,ymax=1.2*v_prev)

        # plt.legend()
        # plt.show()


        ###timestamp plot
        # fig, ax1 = plt.subplots()
        # ax2 = ax1.twinx()
        # ax1.plot(lam,speed, 'g-', label='Speed')
        # ax2.plot(lam, np.gradient(timestamp[start_idx:end_idx+1]), 'b-',label='Timestamp')
        # ax1.set_xlabel('lambda (mm)')
        # ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
        # ax2.set_ylabel('timestamp (s)', color='b')
        # plt.title("Speed: "+data_dir+'v_'+str(v_prev)+'_z10')
        # ax1.legend(loc=0)
        # ax2.legend(loc=0)
        # plt.legend()
        # plt.show()

    
    start_idx=np.argmin(np.linalg.norm(curve[0,:3]-curve_exe,axis=1))
    end_idx=np.argmin(np.linalg.norm(curve[-1,:3]-curve_exe,axis=1))

    curve_exe=curve_exe[start_idx:end_idx+1]
    curve_exe_R=curve_exe_R[start_idx:end_idx+1]
    speed=speed[start_idx:end_idx+1]
    lam=calc_lam_cs(curve_exe)

    ########################################plot verification##################################
    # fig, ax1 = plt.subplots()

    # ax2 = ax1.twinx()
    # ax1.plot(lam,speed, 'g-', label='Speed')
    # ax2.plot(lam, error, 'b-',label='Error')
    # ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')

    # ax1.set_xlabel('lambda (mm)')
    # ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
    # ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
    # plt.title("Speed: "+data_dir+'v_'+str(v)+'_z10')
    # ax1.legend(loc=0)

    # ax2.legend(loc=0)

    # plt.legend()
    # plt.show()

    ######################################save <1mm logged data##############################################
    speed[start_idx:end_idx+1]
    df=DataFrame({'cmd speed':v,'average speed':[np.average(speed)],'max speed':[np.amax(speed)],'min speed':[np.amin(speed)],'std speed':[np.std(speed)],\
        'average error':[np.average(error)],'max error':[max_error],'min error':[np.amin(error)],'std error':[np.std(error)],\
        'average angle error':[np.average(angle_error)],'max angle error':[max(angle_error)],'min angle error':[np.amin(angle_error)],'std angle error':[np.std(angle_error)]})

    df.to_csv(data_dir+'speed_info.csv',header=True,index=False)

    f = open(data_dir+"curve_exe"+"_v"+str(v)+"_z10.csv", "w")
    f.write(logged_data)
    f.close()

if __name__ == "__main__":
    main()