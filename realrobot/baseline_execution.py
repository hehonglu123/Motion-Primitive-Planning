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
sys.path.append('../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from io import StringIO
from lambda_calc import *

def main():
    ms = MotionSend(url='http://192.168.55.1:80')
    robot=abb6640(d=50)

    dataset="from_NX/"
    data_dir='../data/'+dataset+'baseline/100L/'

    curve = read_csv('../data/'+dataset+"Curve_in_base_frame.csv",header=None).values
    breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd(data_dir+'command.csv')
    p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp)

   
    error_threshold=0.5
    angle_threshold=np.radians(3)

    vmax = speeddata(10000,9999999,9999999,999999)
    v=400
    v_prev=2*v
    v_prev_possible=100
    

    max_error=999
    while True:
        ms = MotionSend(url='http://192.168.55.1:80')
        ###update velocity profile
        v_cmd = speeddata(v,9999999,9999999,999999)
        ###5 run execute
        curve_exe_all=[]
        curve_exe_js_all=[]
        timestamp_all=[]
        total_time_all=[]

        for r in range(5):
            logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,v_cmd,z10)

            StringData=StringIO(logged_data)
            df = read_csv(StringData, sep =",")
            ##############################data analysis#####################################
            lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)

            ###throw bad curves
            _, _, _,_, _, timestamp_temp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])
            total_time_all.append(timestamp_temp[-1]-timestamp_temp[0])

            ###TODO, avoid corner path failure

            timestamp=timestamp-timestamp[0]

            curve_exe_all.append(curve_exe)
            curve_exe_js_all.append(curve_exe_js)
            timestamp_all.append(timestamp)

            f = open(data_dir+"realrobot/bisect/run_"+str(r)+"realrobot_curve_exe"+"_v"+str(v)+"_z10.csv", "w")
            f.write(logged_data)
            f.close()

        ###trajectory outlier detection, based on chopped time
        curve_exe_all,curve_exe_js_all,timestamp_all=remove_traj_outlier(curve_exe_all,curve_exe_js_all,timestamp_all,total_time_all)

        ###infer average curve from linear interplateion
        curve_js_all_new, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)
        ###calculat data with average curve
        lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp_d,avg_curve_js)
        #############################chop extension off##################################
        lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp_d,curve[0,:3],curve[-1,:3])

        ##############################calcualte error########################################
        error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
        max_error=np.max(error)
        print('cmd speed: ',v, 'max error: ',max_error, 'max ori error: ', max(angle_error), 'std(speed): ',np.std(speed), 'avg(speed): ',np.average(speed))
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
            ###5 run execute
            curve_exe_all=[]
            curve_exe_js_all=[]
            timestamp_all=[]
            total_time_all=[]

            for r in range(5):
                logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,v_cmd,z10)
                ###save 5 runs
                # Write log csv to file
                with open(path+'/run_'+str(r)+'.csv',"w") as f:
                    f.write(logged_data)

                StringData=StringIO(logged_data)
                df = read_csv(StringData, sep =",")
                ##############################data analysis#####################################
                lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)

                ###throw bad curves
                _, _, _,_, _, timestamp_temp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])
                total_time_all.append(timestamp_temp[-1]-timestamp_temp[0])

                ###TODO, avoid corner path failure

                timestamp=timestamp-timestamp[0]

                curve_exe_all.append(curve_exe)
                curve_exe_js_all.append(curve_exe_js)
                timestamp_all.append(timestamp)

                f = open(data_dir+"run_"+str(r)+"realrobot_curve_exe"+"_v"+str(v)+"_z10.csv", "w")
                f.write(logged_data)
                f.close()

            ###trajectory outlier detection, based on chopped time
            curve_exe_all,curve_exe_js_all,timestamp_all=remove_traj_outlier(curve_exe_all,curve_exe_js_all,timestamp_all,total_time_all)

            ###infer average curve from linear interplateion
            curve_js_all_new, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)
            ###calculat data with average curve
            lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp_d,avg_curve_js)
            #############################chop extension off##################################
            lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp_d,curve[0,:3],curve[-1,:3])

            ##############################calcualte error########################################
            error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
            
            max_error=max(error)
            break

    ######################################save <1mm logged data##############################################
    df=DataFrame({'cmd speed':v,'average speed':[np.average(speed)],'max speed':[np.amax(speed)],'min speed':[np.amin(speed)],'std speed':[np.std(speed)],\
        'average error':[np.average(error)],'max error':[max_error],'min error':[np.amin(error)],'std error':[np.std(error)],\
        'average angle(rad) error':[np.average(angle_error)],'max angle(rad) error':[max(angle_error)],'min angle(rad) error':[np.amin(angle_error)],'std angle(rad) error':[np.std(angle_error)]})

    df.to_csv(data_dir+'realrobot/bisect/speed_info.csv',header=True,index=False)



if __name__ == "__main__":
    main()