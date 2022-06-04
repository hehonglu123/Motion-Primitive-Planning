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
from blending import *

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
    inserted_points=[]
    while max_error>max_error_threshold:
        ms = MotionSend()
        ###execute,curve_fit_js only used for orientation
        logged_data=ms.exec_motions(robot,primitives,breakpoints,points_list,curve_fit_js,s,z)

        StringData=StringIO(logged_data)
        df = read_csv(StringData, sep =",")
        ##############################data analysis#####################################
        lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df)
        #############################chop extension off##################################
        start_idx=np.argmin(np.linalg.norm(curve[0,:3]-curve_exe,axis=1))
        end_idx=np.argmin(np.linalg.norm(curve[-1,:3]-curve_exe,axis=1))

        #make sure extension doesn't introduce error
        if np.linalg.norm(curve_exe[start_idx]-curve[0,:3])>0.5:
            start_idx+=1
        if np.linalg.norm(curve_exe[end_idx]-curve[-1,:3])>0.5:
            end_idx-=1

        curve_exe=curve_exe[start_idx:end_idx+1]
        curve_exe_R=curve_exe_R[start_idx:end_idx+1]
        speed=speed[start_idx:end_idx+1]
        speed=replace_outliers(np.array(speed))
        lam=calc_lam_cs(curve_exe)

        ##############################calcualte error########################################
        error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
        max_error=max(error)
        print(max_error)
        max_angle_error=max(angle_error)
        max_error_idx=np.argmax(error)#index of exe curve with max error
        _,max_error_curve_idx=calc_error(curve_exe[max_error_idx],curve[:,:3])  # index of original curve closest to max error point
        exe_error_vector=(curve[max_error_curve_idx,:3]-curve_exe[max_error_idx])           # shift vector
        
        ##############################plot error#####################################
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(lam, speed, 'g-', label='Speed')
        ax2.plot(lam, error, 'b-',label='Error')
        ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')

        ax1.set_xlabel('lambda (mm)')
        ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
        ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
        plt.title("Speed and Error Plot")
        ax1.legend(loc=0)

        ax2.legend(loc=0)

        plt.legend()
        ###########################find closest bp####################################
        bp_idx=np.absolute(breakpoints-max_error_curve_idx).argmin()
        ###########################plot for verification###################################
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='gray',label='original')
        ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c='red',label='execution')
        points_list_np=np.array(points_list[1:-1])      ###np version, avoid first and last extended points
        ax.scatter3D(points_list_np[:,0], points_list_np[:,1], points_list_np[:,2], c=points_list_np[:,2], cmap='Greens',label='breakpoints')
        ax.scatter(curve_exe[max_error_idx,0], curve_exe[max_error_idx,1], curve_exe[max_error_idx,2],c='orange',label='worst case')
        
        
        ##########################################calculate gradient######################################
        ######gradient calculation related to nearest 3 points from primitive blended trajectory, not actual one
        ###restore trajectory from primitives
        q_bp=[]
        for i in range(len(primitives)):
            if primitives[i]=='movej_fit':
                q_bp.append(points_list[i])
            elif primitives[i]=='movel_fit':

                q_bp.append(car2js(robot,curve_fit_js[breakpoints[i]],np.array(points_list[i]),robot.fwd(curve_fit_js[breakpoints[i]]).R)[0])
            else:
                q_bp.append([car2js(robot,curve_fit_js[int((breakpoints[i]+breakpoints[i-1])/2)],points_list[i][0],robot.fwd(curve_fit_js[int((breakpoints[i]+breakpoints[i-1])/2)]).R)[0]\
                    ,car2js(robot,curve_fit_js[breakpoints[i]],points_list[i][0],robot.fwd(curve_fit_js[breakpoints[i]]).R)[0]])


        curve_interp, curve_R_interp, curve_js_interp, breakpoints_interp=form_traj_from_bp(q_bp,primitives,robot)

        curve_js_blended,curve_blended,curve_R_blended=blend_js_from_primitive(curve_interp, curve_js_interp, breakpoints_interp, primitives,robot,zone=10)

        _,max_error_curve_blended_idx=calc_error(curve_exe[max_error_idx],curve_blended)
        curve_blended_point=copy.deepcopy(curve_blended[max_error_curve_blended_idx])

        ###############get numerical gradient#####
        de_dp=[]    #de_dp1x,de_dp1y,...,de_dp3z
        delta=0.1#mm
        ###find closest 3 breakpoints
        order=np.argsort(np.abs(breakpoints_interp-max_error_curve_blended_idx))
        breakpoint_interp_2tweak_indices=order[:3]

        #len(primitives)==len(breakpoints)==len(breakpoints_interp)==len(points_list)
        for m in breakpoint_interp_2tweak_indices:  #3 breakpoints
            for n in range(3): #3DOF, xyz
                points_list_temp=copy.deepcopy(points_list)
                q_bp_temp=copy.deepcopy(q_bp)
                points_list_temp[m][n]+=delta
                q_bp_temp[m]=car2js(robot,q_bp_temp[m],np.array(points_list_temp[m]),robot.fwd(q_bp_temp[m]).R)[0]
                #restore new trajectory
                curve_interp_temp, curve_R_interp_temp, curve_js_interp_temp, breakpoints_interp_temp=form_traj_from_bp(q_bp_temp,primitives,robot)
                curve_js_blended_temp,curve_blended_temp,curve_R_blended_temp=blend_js_from_primitive(curve_interp_temp, curve_js_interp_temp, breakpoints_interp_temp, primitives,robot,zone=10)
                
                worst_case_point_shift=curve_blended_temp[max_error_curve_blended_idx]-curve_blended[max_error_curve_blended_idx]
                de=np.linalg.norm(exe_error_vector-worst_case_point_shift)-max_error
                de_dp.append(de/delta)


        de_dp=np.reshape(de_dp,(-1,1))
        alpha=0.5
        point_adjustment=-alpha*np.linalg.pinv(de_dp)*max_error

        for i in range(len(breakpoint_interp_2tweak_indices)):  #3 breakpoints
            points_list[breakpoint_interp_2tweak_indices[i]]+=point_adjustment[0][3*i:3*(i+1)]

        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()