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
    #determine correct commanded speed to keep error within 1mm
    thresholds=[0.5]
    max_error_threshold=0.5
    max_ori_threshold=np.radians(3)

    dataset='wood/'
    data_dir="../../../data/"+dataset
    

    curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values

    for threshold in thresholds:
        ms = MotionSend()
        
        fitting_output="fitting_output/"+dataset+'threshold'+str(threshold)+'/'

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
            logged_data=ms.exe_from_file_w_extension(robot,fitting_output+"command.csv",fitting_output+"curve_fit_js.csv",v_cmd,z10,extension_d=50)
            StringData=StringIO(logged_data)
            df = read_csv(StringData, sep =",")
            ##############################data analysis#####################################
            lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df)
            error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:],extension=True)
            
            max_error=max(error)
            max_angle_error=max(angle_error)
            print(v,max_error)
            v_prev_temp=v
            if max_error>max_error_threshold or max_angle_error>max_ori_threshold:
                v-=abs(v_prev-v)/2
            else:
                v_prev_possible=v
                #stop condition
                if max_error_threshold-max_error<max_error_threshold/10 or max_ori_threshold-max_angle_error<max_ori_threshold/10:
                    break   
                v+=abs(v_prev-v)/2

            v_prev=v_prev_temp

            #if stuck
            if abs(v-v_prev)<0.1:
                v=v_prev_possible
                v_cmd = speeddata(v,9999999,9999999,999999)
                logged_data=ms.exe_from_file_w_extension(fitting_output+"command.csv",fitting_output+"curve_fit_js.csv",v_cmd,z10,extension_d=100)
                StringData=StringIO(logged_data)
                df = read_csv(StringData, sep =",")
                lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df)
                error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:],extension=True)
                
                max_error=max(error)
                break

        
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
        # ax2.set_ylabel('Error (mm)', color='b')
        # plt.title("Speed: "+datfitting_outputa_dir+'v_'+str(v)+'_z10')
        # ax1.legend(loc=0)

        # ax2.legend(loc=0)

        # plt.legend()
        # plt.show()

        ######################################save <1mm logged data##############################################
        speed[start_idx:end_idx+1]
        df=DataFrame({'average speed':[np.average(speed)],'max speed':[np.amax(speed)],'min speed':[np.amin(speed)],'std speed':[np.std(speed)],\
            'average error':[np.average(error)],'max error':[max_error],'min error':[np.amin(error)],'std error':[np.std(error)],\
            'average angle error':[np.average(angle_error)],'max angle error':[max(angle_error)],'min angle error':[np.amin(angle_error)],'std angle error':[np.std(angle_error)]})

        df.to_csv(fitting_output+'speed_info.csv',header=True,index=False)

        f = open(fitting_output+"curve_exe"+"_v"+str(v)+"_z10.csv", "w")
        f.write(logged_data)
        f.close()

if __name__ == "__main__":
    main()