import numpy as np
from scipy.signal import find_peaks
from general_robotics_toolbox import *
from pandas import read_csv,DataFrame
import sys
from io import StringIO
from matplotlib import pyplot as plt
from pathlib import Path
import os
import copy
from fanuc_motion_program_exec_client import *

# sys.path.append('../abb_motion_program_exec')
# from abb_motion_program_exec_client import *
sys.path.append('../fanuc_toolbox')
from fanuc_utils import *
sys.path.append('../stream_motion')
from MS_toolbox import *
sys.path.append('../../../ILC')
from ilc_toolbox import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *
from blending import *


def main():
    # data_dir="fitting_output_new/python_qp_movel/"
    dataset='blade/'
    # dataset='wood/'
    data_dir="data/"+dataset
    fitting_output='data/'+dataset

    # robot
    robot=m710ic(d=50)

    curve_js=read_csv(data_dir+'Curve_js.csv',header=None).values
    curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
    curve_normal=curve[:,3:]
    curve_R=[]
    for q in curve_js:
        curve_R.append(robot.fwd(q).R)
    curve_R=np.array(curve_R)

    # FANUC Stream Motion
    ms=MotionStream()
    mst = MS_toolbox(ms,robot)

    # cmd
    # vd=125
    # max_error_threshold=0.1
    # lam=calc_lam_cs(curve[:,:3])
    # steps=int((lam[-1]/vd)/mst.dt)
    # print(steps)
    # breakpoints=np.linspace(0.,len(curve_js)-1,num=steps).astype(int)
    # curve_cmd_js=curve_js[breakpoints]
    # curve_cmd=curve[breakpoints,:3]
    # curve_cmd_R=curve_R[breakpoints]
    
    steps=105
    breakpoints=[]
    curve_cmd_js=curve_js[::steps]
    curve_cmd=curve[::steps,:3]
    curve_cmd_R=curve_R[::steps]
    lam=calc_lam_cs(curve[:,:3])
    vd=int(lam[-1]/((len(curve_cmd_js)-1)*mst.dt))
    print(vd)
    print(len(curve_js))
    print(len(curve_cmd_js))

    # output folder
    ilc_output=fitting_output+'results_ilc_'+str(vd)+'/'
    Path(ilc_output).mkdir(exist_ok=True)
    

    ### fanuc Motion Stream toolbox has built in extension
    q_bp=[]
    p_bp=[]
    for i in range(len(curve_cmd_js)):
        q_bp.append([curve_cmd_js[i]])
        p_bp.append([curve_cmd[i]])
    primitives,p_bp,q_bp=extend_start_end(robot,q_bp,['movej_fit','movej_fit'],breakpoints,p_bp,extension_d=50)
    start_q=q_bp[0][0]
    end_q=q_bp[-1][0]

    curve_cmd_w=R2w(curve_cmd_R)
    curve_d=copy.deepcopy(curve_cmd)
    curve_R_d=copy.deepcopy(curve_cmd_R)
    curve_w_d=copy.deepcopy(curve_cmd_w)

    max_error=999
    iteration=30
    adjust_weigt_it=30
    draw_error_max=None
    draw_speed_max=None
    for i in range(iteration):

        ###move to start first
        print('moving to start point')
        mst.jog_joint(start_q)
        
        ###traverse the curve
        curve_ts,curve_js_exe,curve_js_plan=mst.traverse_js_curve(curve_cmd_js,end_q)

        ##############################data analysis#####################################
        lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=mst.logged_data_analysis(robot,curve_js_exe,curve_ts)

        #############################chop extension off##################################
        lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=mst.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[:,:3],curve[:,3:])
        ave_speed=np.mean(speed)

        ##############################calcualte error########################################
        error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
        print('Iteration:',i,', Max Error:',max(error),'Ave. Speed:',ave_speed,'Std. Speed:',np.std(speed),'Std/Ave (%):',np.std(speed)/ave_speed*100)
        print('Max Speed:',max(speed),'Min Speed:',np.min(speed),'Ave. Error:',np.mean(error),'Min Error:',np.min(error),"Std. Error:",np.std(error))
        print('Max Ang Error:',max(np.degrees(angle_error)),'Min Ang Error:',np.min(np.degrees(angle_error)),'Ave. Ang Error:',np.mean(np.degrees(angle_error)),"Std. Ang Error:",np.std(np.degrees(angle_error)))
        print("===========================================")

        print(np.linalg.norm(curve_exe[-1]-curve[-1,:3]))

        ##############################plot error#####################################
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(lam, speed, 'g-', label='Speed')
        ax2.plot(lam, error, 'b-',label='Error')
        ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
        if draw_error_max is None:
            draw_error_max=max([max(error)*1.05,0.5])
        if draw_speed_max is None:
            draw_speed_max=max([max(speed)*1.05,100])
        # draw_y_max=2
        ax1.axis(ymin=0,ymax=draw_speed_max)
        ax2.axis(ymin=0,ymax=draw_error_max)
        ax1.set_xlabel('lambda (mm)')
        ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
        ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
        plt.title("Speed and Error Plot")
        ax1.legend(loc=0)
        ax2.legend(loc=0)

        # save fig
        plt.legend()
        # plt.savefig(ilc_output+'iteration_'+str(i))
        # plt.clf()
        plt.show()

        fig, axs = plt.subplots(2, 3)
        timestamp_plan = np.linspace(0,(len(curve_js_plan)-1)*mst.dt,len(curve_js_plan))
        for j in range(6):
            axs[int(j/3),j%3].plot(timestamp_plan,curve_js_plan[:,j],label='Plan')
            axs[int(j/3),j%3].plot(curve_ts/1000.,curve_js_exe[:,j],label='Exe')
            axs[int(j/3),j%3].set_title('Joint '+str(j+1))
            axs[int(j/3),j%3].legend(loc=0)
        plt.legend()
        plt.show()

        exit()
        curve_exe_w=R2w(curve_exe_R,curve_R_d[0])
        ##############################ILC########################################
        error=curve_exe-curve_d
        error_distance=np.linalg.norm(error,axis=1)
        print('worst case error: ',np.max(error_distance))
        ##add weights based on error
        weights_p=np.ones(len(error))
        # weights_p=np.linalg.norm(error,axis=1)
        # weights_p=(len(error)/4)*weights_p/weights_p.sum()
        if i>adjust_weigt_it:
            weights_p[np.where(error_distance>0.5*np.max(error_distance))]=5

            

        error=error*weights_p[:, np.newaxis]
        error_flip=np.flipud(error)
        error_w=curve_exe_w-curve_w_d
        #add weights based on error_w
        # weights_w=np.linalg.norm(error_w,axis=1)
        # weights_w=(len(error)/4)*weights_w/weights_w.sum()
        weights_w=np.ones(len(error_w))

        error_w=error_w*weights_w[:, np.newaxis]
        error_w_flip=np.flipud(error_w)
        ###calcualte agumented input
        curve_cmd_aug=curve_cmd+error_flip
        curve_cmd_w_aug=curve_cmd_w+error_w_flip
        curve_cmd_R_aug=w2R(curve_cmd_w_aug,curve_R_d[0])


        

        ###add extension
        curve_cmd_ext_aug,curve_cmd_R_ext_aug=et.add_extension_egm_cartesian(curve_cmd_aug,curve_cmd_R_aug,extension_num=extension_num)
        ###move to start first
        print('moving to start point')
        et.jog_joint_cartesian(curve_cmd_ext_aug[0],curve_cmd_R_ext_aug[0])
        ###traverse the curve
        timestamp_aug,curve_exe_js_aug=et.traverse_curve_cartesian(curve_cmd_ext_aug,curve_cmd_R_ext_aug)

        _, curve_exe_aug, curve_exe_R_aug, _=logged_data_analysis(robot,timestamp_aug[extension_num+idx_delay:-extension_num+idx_delay],curve_exe_js_aug[extension_num+idx_delay:-extension_num+idx_delay])

        ###get new error
        delta_new=curve_exe_aug-curve_exe
        delta_w_new=R2w(curve_exe_R_aug,curve_R_d[0])-curve_exe_w

        grad=np.flipud(delta_new)
        grad_w=np.flipud(delta_w_new)

        alpha1=1-i/iteration#0.5
        alpha2=1-i/iteration#1
        curve_cmd_new=curve_cmd-alpha1*grad
        curve_cmd_w-=alpha2*grad_w
        curve_cmd_R=w2R(curve_cmd_w,curve_R_d[0])

        curve_cmd=curve_cmd_new

    ##############################plot error#####################################
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(lam[1:], speed, 'g-', label='Speed')
    ax2.plot(lam, error_distance, 'b-',label='Error')
    ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')

    ax1.set_xlabel('lambda (mm)')
    ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
    ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
    plt.title("Speed and Error Plot")
    ax1.legend(loc=0)

    ax2.legend(loc=0)

    plt.legend()

    ###########################plot for verification###################################
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='gray',label='original')
    ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c='red',label='execution')
    ax.scatter3D(curve_cmd[:,0], curve_cmd[:,1], curve_cmd[:,2], c=curve_cmd[:,2], cmap='Greens',label='commanded points')
    ax.scatter3D(curve_cmd_new[:,0], curve_cmd_new[:,1], curve_cmd_new[:,2], c=curve_cmd_new[:,2], cmap='Blues',label='new commanded points')


    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()