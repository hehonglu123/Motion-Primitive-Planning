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
    data_dir="../../../train_data/"+dataset


    curve_js=read_csv(data_dir+'Curve_js.csv',header=None).values
    curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values


    max_error_threshold=0.5
    robot=abb6640(d=50)

    s = speeddata(150,9999999,9999999,999999)
    z = z10

    fitting_output="../../../train_data/"+dataset+'baseline/100L/'
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
        ###execute
        logged_data=ms.exec_motions(robot,primitives,breakpoints,points_list,curve_fit_js,s,z)

        StringData=StringIO(logged_data)
        df = read_csv(StringData, sep =",")
        ##############################train_data analysis#####################################
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
        d=(curve[max_error_curve_idx,:3]-curve_exe[max_error_idx])#/2           # shift vector
        
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
        
        

        
        ##############################Tweak breakpoints#################################################
        closest_bp = breakpoints[bp_idx]
        print(closest_bp,max_error_curve_idx)
        if abs(closest_bp-max_error_curve_idx)<150:  ###arb value, to decide insert or tweak existing bp
            print('adjusting breakpoints')
            points_list[bp_idx]+=d

            ax.quiver(points_list_np[bp_idx-1,0],points_list_np[bp_idx-1,1],points_list_np[bp_idx-1,2],d[0],d[1],d[2],length=1, normalize=True)
        else:
            print('inserting breakpoints')
            if breakpoints[bp_idx]>max_error_curve_idx:
                insertion_idx=bp_idx            #insertion index, the later one
            else:
                insertion_idx=bp_idx+1
            points_list.insert(insertion_idx,curve[max_error_curve_idx,:3]+d)
            breakpoints=np.insert(breakpoints,insertion_idx,max_error_curve_idx)
            primitives.insert(insertion_idx,'movel_fit')
            inserted_points.append(curve[max_error_curve_idx,:3]+d)

            ax.scatter(inserted_points[-1][0],inserted_points[-1][1],inserted_points[-1][2],c='blue',label='inserted point')

        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()