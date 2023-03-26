import numpy as np
from general_robotics_toolbox import *
import sys
sys.path.append('../../toolbox')
from robots_def import *
from error_check import *
from realrobot import *
from MotionSend_motoman import *
from lambda_calc import *

def main():
    robot=robot_obj('MA2010_A0',def_path='../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun2.csv',\
    pulse2deg_file_path='../../config/MA2010_A0_pulse2deg.csv',d=50)
    ms = MotionSend(robot)

    dataset='curve_1/'
    solution_dir='baseline_motoman/'
    data_dir='../../data/'+dataset+solution_dir
    cmd_dir=data_dir+'50L/'

    curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
    breakpoints,primitives, p_bp,q_bp=ms.extract_data_from_cmd(cmd_dir+"command.csv")
    p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp)

   
    error_threshold=0.5
    angle_threshold=np.radians(3)

    v=400
    v_prev=2*v
    v_prev_possible=100
    z=None

    N=5
    

    max_error=999
    while True:
        ms = MotionSend(robot)

        curve_js_all_new, avg_curve_js, timestamp_d=average_N_exe(ms,robot,primitives,breakpoints,p_bp,q_bp,v,z,curve,"recorded_data",N=N)
        ###calculat data with average curve
        lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp_d,avg_curve_js)
        #############################chop extension off##################################
        lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,avg_curve_js, speed, timestamp_d,curve[0,:3],curve[-1,:3])

        ms.write_data_to_cmd('recorded_data/command.csv',breakpoints,primitives, p_bp,q_bp)

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
            ms = MotionSend(robot)
            curve_js_all_new, avg_curve_js, timestamp_d=average_N_exe(ms,robot,primitives,breakpoints,p_bp,q_bp,s,z,curve,"recorded_data",N=N)
            ###calculat data with average curve
            lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp_d,avg_curve_js)
            #############################chop extension off##################################
            lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,avg_curve_js, speed, timestamp_d,curve[0,:3],curve[-1,:3])

            ms.write_data_to_cmd('recorded_data/command.csv',breakpoints,primitives, p_bp,q_bp)

            ##############################calcualte error########################################
            error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
            max_error=np.max(error)
            break

    ######################################save <1mm logged train_data##############################################
    df=DataFrame({'cmd speed':v,'average speed':[np.average(speed)],'max speed':[np.amax(speed)],'min speed':[np.amin(speed)],'std speed':[np.std(speed)],\
        'average error':[np.average(error)],'max error':[max_error],'min error':[np.amin(error)],'std error':[np.std(error)],\
        'average angle(rad) error':[np.average(angle_error)],'max angle(rad) error':[max(angle_error)],'min angle(rad) error':[np.amin(angle_error)],'std angle(rad) error':[np.std(angle_error)]})

    df.to_csv(data_dir+'realrobot/bisect/speed_info.csv',header=True,index=False)



if __name__ == "__main__":
    main()