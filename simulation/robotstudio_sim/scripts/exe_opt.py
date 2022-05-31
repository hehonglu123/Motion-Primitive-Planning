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
from lambda_calc import *

def main():
    # data_dir="fitting_output_new/python_qp_movel/"
    dataset='wood/'
    data_dir="../../../data/"+dataset


    curve_js=read_csv(data_dir+'Curve_js.csv',header=None).values
    curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values


    max_error_threshold=0.5
    robot=abb6640(d=50)

    s = speeddata(150,9999999,9999999,999999)
    z = z10

    fitting_output="../../../data/"+dataset+'baseline/100L/'
    data = read_csv(fitting_output+'command.csv')
    breakpoints=np.array(data['breakpoints'].tolist())
    primitives=data['primitives'].tolist()
    points=data['points'].tolist()
    
    breakpoints[1:]=breakpoints[1:]-1
    curve_fit_js=read_csv(fitting_output+'curve_fit_js.csv',header=None).values

    points_list=[]
    for i in range(len(breakpoints)):
        if primitives[i]=='movel_fit':
            point=extract_points(primitives[i],points[i])
            points_list.append(point)
        elif primitives[i]=='movec_fit':
            point1,point2=extract_points(primitives[i],points[i])
            points_list.append([point1,point2])
        else:
            point=extract_points(primitives[i],points[i])
            points_list.append(point)

    ms = MotionSend()
        
    ###extension
    primitives,points_list,curve_fit_js=ms.extend(robot,curve_fit_js,primitives,breakpoints,points_list)

    ###TODO: extension fix start point, moveC support
    max_error=999
    while max_error>max_error_threshold:
        ms = MotionSend()
        ###execute
        logged_data=ms.exec_motions(robot,primitives,breakpoints,points_list,curve_fit_js,s,z)

        StringData=StringIO(logged_data)
        df = read_csv(StringData, sep =",")
        ##############################data analysis#####################################
        lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df)
        error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:],extension=True)
        max_error=max(error)
        print(max_error)
        max_angle_error=max(angle_error)
        max_error_idx=np.argmax(error)#index of exe curve with max error
        _,max_error_curve_idx=calc_error(curve_exe[max_error_idx],curve[:,:3])  # index of original curve closest to max error point
        d=(curve[max_error_curve_idx,:3]-curve_exe[max_error_idx])/2           # shift vector
        #############################chop extension off##################################
        start_idx=np.argmin(np.linalg.norm(curve[0,:3]-curve_exe,axis=1))
        end_idx=np.argmin(np.linalg.norm(curve[-1,:3]-curve_exe,axis=1))

        curve_exe=curve_exe[start_idx:end_idx+1]
        curve_exe_R=curve_exe_R[start_idx:end_idx+1]
        speed=speed[start_idx:end_idx+1]
        lam=calc_lam_cs(curve_exe)
        ###########################plot for verification###################################
        plt.figure()
        plt.plot(lam,error)
        plt.show()
        ##############################Tweak breakpoints#################################################
        closest_bp = breakpoints[np.absolute(breakpoints-max_error_curve_idx).argmin()]
        print(closest_bp,max_error_curve_idx)
        if abs(closest_bp-max_error_curve_idx)<50:  ###arb value, to decide insert or tweak existing bp
            print('adjusting breakpoints')
            points_list[np.where(breakpoints==closest_bp)[0][0]]+=d





if __name__ == "__main__":
    main()