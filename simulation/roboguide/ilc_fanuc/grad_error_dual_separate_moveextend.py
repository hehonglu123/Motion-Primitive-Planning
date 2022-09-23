########
# This module utilized https://github.com/eric565648/fanuc_motion_program_exec_client
# and send whatever the motion primitives that algorithms generate
# to Roboguide
########

from copy import deepcopy
import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv,DataFrame
import sys
from io import StringIO
from scipy.signal import find_peaks
import yaml
from matplotlib import pyplot as plt
from pathlib import Path
from fanuc_motion_program_exec_client import *
sys.path.append('../fanuc_toolbox')
from fanuc_utils import *
sys.path.append('../../../ILC')
from ilc_toolbox import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *
from blending import *

def error_speed_plot(lam,speed,error,peaks,angle_error,draw_speed_max,draw_error_max,output_folder,output_filename,save_fig=False):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(lam, speed, 'g-', label='Speed')
    ax2.plot(lam, error, '-bo', markersize=2,label='Error')
    # ax2.scatter(lam, error, 'b-',label='Error')
    ax2.scatter(lam[peaks],error[peaks],label='peaks')
    ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
    ax1.axis(ymin=0,ymax=draw_speed_max)
    ax2.axis(ymin=0,ymax=draw_error_max)
    ax1.set_xlabel('lambda (mm)')
    ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
    ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
    plt.title("Speed and Error Plot")
    ax1.legend(loc=0)
    ax2.legend(loc=0)
    plt.legend()
    if save_fig:
        plt.savefig(output_folder+output_filename)
        plt.clf()
    else:
        plt.show()

def main():
    # curve
    # data_type='blade'
    # data_type='wood'
    # data_type='blade_arm_shift'
    data_type='blade_base_shift'
    # data_type='wood_base_shift'

    # data and curve directory
    if data_type=='blade':
        curve_data_dir='../../../data/from_NX/'
        cmd_dir='../data/curve_blade/'
        data_dir='data/blade_dual/'
    elif data_type=='wood':
        curve_data_dir='../../../data/wood/'
        cmd_dir='../data/curve_wood/'
        data_dir='data/wood_dual/'
    elif data_type=='blade_arm_shift':
        curve_data_dir='../../../data/from_NX/'
        cmd_dir='../data/curve_blade_arm_shift/'
        data_dir='data/blade_arm_shift_dual/'
    elif data_type=='blade_base_shift':
        curve_data_dir='../../../data/from_NX/'
        cmd_dir='../data/curve_blade_base_shift/'
        data_dir='data/blade_base_shift_dual/'
    elif data_type=='wood_base_shift':
        curve_data_dir='../../../data/wood/'
        cmd_dir='../data/curve_wood_base_shift/'
        data_dir='data/wood_base_shift_dual/'

    test_type='dual_arm'
    # test_type='dual_single_arm'
    # test_type='dual_single_arm_straight' # robot2 is multiple user defined straight line
    # test_type='dual_single_arm_straight_50' # robot2 is multiple user defined straight line
    # test_type='dual_single_arm_straight_min' # robot2 is multiple user defined straight line
    # test_type='dual_single_arm_straight_min10' # robot2 is multiple user defined straight line
    # test_type='dual_arm_10'
    # test_type='dual_arm_qp'

    data_dir=data_dir[:-1]+'_2ctrller/'

    cmd_dir=cmd_dir+test_type+'/'

    # relative path
    relative_path = read_csv(curve_data_dir+"/Curve_dense.csv", header=None).values

    # the second robot relative to the fist robot
    with open(cmd_dir+'../m900ia.yaml') as file:
        H_robot2 = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
    base2_R=H_robot2[:3,:3]
    base2_p=1000*H_robot2[:-1,-1]
    base2_T=rox.Transform(base2_R,base2_p)
    # workpiece (curve) relative to robot tcp
    with open(cmd_dir+'../tcp.yaml') as file:
        H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
    
    # define robot
    robot1=m710ic(d=50)
    robot2=m900ia(R_tool=H_tcp[:3,:3],p_tool=H_tcp[:-1,-1])
    # robot2=m900ia(R_tool=np.matmul(Ry(np.radians(90)),Rz(np.radians(180))),p_tool=np.array([0,0,0])*1000.,d=0)

    # arm path
    curve_js1 = read_csv(cmd_dir+"/arm1.csv", header=None).values
    curve_js2 = read_csv(cmd_dir+"/arm2.csv", header=None).values
    try:
        curve1 = np.array(read_csv(cmd_dir+"/arm1_curve.csv", header=None).values)
        curve2 = np.array(read_csv(cmd_dir+"/arm2_curve.csv", header=None).values)
        curve_n1 = curve1[:,3:]
        curve1 = curve1[:,:3]
        curve_n2 = curve2[:,3:]
        curve2 = curve2[:,:3]
    except:
        curve1 = []
        curve_R1 = []
        curve_n1 = []
        curve2 = []
        curve_R2 = []
        curve_n2 = []
        for i in range(len(curve_js1)):
            curve_T1 = robot1.fwd(curve_js1[i])
            curve_T2 = robot2.fwd(curve_js2[i])
            curve1.append(curve_T1.p)
            curve2.append(curve_T2.p)
            curve_R1.append(curve_T1.R)
            curve_n1.append(curve_T1.R[:,-1])
            curve_R2.append(curve_T2.R)
            curve_n2.append(curve_T2.R[:,-1])
        curve1 = np.array(curve1)
        curve2 = np.array(curve2)
        curve_n1 = np.array(curve_n1)
        curve_n2 = np.array(curve_n2)
        df=DataFrame({'x':curve1[:,0],'y':curve1[:,1],'z':curve1[:,2],'nx':curve_n1[:,0],'ny':curve_n1[:,1],'nz':curve_n1[:,2]})
        df.to_csv(cmd_dir+'/arm1_curve.csv',header=False,index=False)
        df=DataFrame({'x':curve2[:,0],'y':curve2[:,1],'z':curve2[:,2],'nx':curve_n2[:,0],'ny':curve_n2[:,1],'nz':curve_n2[:,2]})
        df.to_csv(cmd_dir+'/arm2_curve.csv',header=False,index=False)

    # fanuc motion send tool
    if data_type=='blade':
        ms = MotionSendFANUC(robot1=robot1,robot2=robot2,group2=1,robot_ip2='127.0.0.3')
    elif data_type=='wood':
        ms = MotionSendFANUC(robot1=robot1,robot2=robot2,group2=1,utool2=3,robot_ip2='127.0.0.3')
    elif data_type=='blade_arm_shift':
        ms = MotionSendFANUC(robot1=robot1,robot2=robot2,group2=1,utool2=6,robot_ip2='127.0.0.3')
    elif data_type=='blade_base_shift':
        ms = MotionSendFANUC(robot1=robot1,robot2=robot2,group2=1,utool2=2,robot_ip2='127.0.0.3')
    elif data_type=='wood_base_shift':
        ms = MotionSendFANUC(robot1=robot1,robot2=robot2,group2=1,utool2=3,robot_ip2='127.0.0.3')

    # s=int(1600/2.) # mm/sec in leader frame
    s=1000 # mm/sec
    # s=500 # mm/sec
    # s=16 # mm/sec in leader frame
    z=100 # CNT100
    ilc_output=data_dir+'results_'+str(s)+'_'+test_type+'/'
    Path(ilc_output).mkdir(exist_ok=True)

    breakpoints1,primitives1,p_bp1,q_bp1,_=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command1.csv')
    breakpoints2,primitives2,p_bp2,q_bp2,_=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command2.csv')

    ### calculate speed of each segments
    p_bp_relative,_=ms.form_relative_path(np.squeeze(q_bp1),np.squeeze(q_bp2),base2_R,base2_p)
    dlam_movel=np.linalg.norm(np.diff(p_bp_relative,axis=0),2,1)
    dt_movel = dlam_movel/s
    p_bp1_sq = np.squeeze(p_bp1)
    p_bp2_sq = np.squeeze(p_bp2)
    dlam1_movel=np.linalg.norm(np.diff(p_bp1_sq,axis=0),2,1)
    dlam2_movel=np.linalg.norm(np.diff(p_bp2_sq,axis=0),2,1)
    s1_movel = np.divide(dlam1_movel,dt_movel)
    s2_movel = np.divide(dlam2_movel,dt_movel)
    s1_movel = np.append(s1_movel[0],s1_movel)
    s2_movel = np.append(s2_movel[0],s2_movel)

    q_bp1_start = q_bp1[0][0]
    q_bp1_end = q_bp1[-1][-1]
    q_bp2_start = q_bp2[0][0]
    q_bp2_end = q_bp2[-1][-1]

    p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1,base2_T,extension_d=70)

    ## calculate step at start and end
    step_start1=None
    step_end1=None
    for i in range(len(q_bp1)):
        if np.all(q_bp1[i][0]==q_bp1_start):
            step_start1=i
        if np.all(q_bp1[i][-1]==q_bp1_end):
            step_end1=i

    assert step_start1 is not None,'Cant find step start'
    assert step_end1 is not None,'Cant find step end'
    print(step_start1,step_end1)

    step_start2=None
    step_end2=None
    for i in range(len(q_bp2)):
        if np.all(q_bp2[i][0]==q_bp2_start):
            step_start2=i
        if np.all(q_bp2[i][-1]==q_bp2_end):
            step_end2=i

    assert step_start2 is not None,'Cant find step start'
    assert step_end2 is not None,'Cant find step end'
    print(step_start2,step_end2)


    def dummy_loop(the_item,step_start,step_end):
        the_item_dum = [the_item[0]]
        for i in range(step_start,step_end+1):
            the_item_dum.append(the_item[i])
        the_item_dum.append(the_item[-1])
        return the_item_dum
    p_bp1 = dummy_loop(p_bp1,step_start1,step_end1)
    q_bp1 = dummy_loop(q_bp1,step_start1,step_end1)
    primitives1 = dummy_loop(primitives1,step_start1,step_end1)
    p_bp2 = dummy_loop(p_bp2,step_start2,step_end2)
    q_bp2 = dummy_loop(q_bp2,step_start2,step_end2)
    primitives2 = dummy_loop(primitives2,step_start2,step_end2)
    step_start1,step_start2,step_end1,step_end2=1,1,len(p_bp1)-2,len(p_bp2)-2

    ## extend speed
    s1_start = np.ones(step_start1)*s1_movel[0]
    s2_start = np.ones(step_start2)*s2_movel[0]
    s1_end = np.ones(len(p_bp1)-step_end1-1)*s1_movel[-1]
    s2_end = np.ones(len(p_bp2)-step_end2-1)*s2_movel[-1]
    s1_movel=np.append(np.append(s1_start,s1_movel),s1_end)
    s2_movel=np.append(np.append(s2_start,s2_movel),s2_end)

    print(len(p_bp1),len(q_bp1),len(primitives1),len(s1_movel))
    print(len(p_bp2),len(q_bp2),len(primitives2),len(s2_movel))

    ## round up speed
    s1_movel = np.around(s1_movel)
    s2_movel = np.around(s2_movel)
    s1_movel_des = deepcopy(s1_movel)
    s2_movel_des = deepcopy(s2_movel)

    use_exist=False
    use_iteration=23
    if use_exist:
        _,primitives1,p_bp1,q_bp1,s1_movel=ms.extract_data_from_cmd(os.getcwd()+'/'+ilc_output+'command_arm1_'+str(use_iteration)+'.csv')
        _,primitives2,p_bp2,q_bp2,s2_movel=ms.extract_data_from_cmd(os.getcwd()+'/'+ilc_output+'command_arm2_'+str(use_iteration)+'.csv')

    # for i in range(len(primitives1)):
    #     primitives1[i]='movej_fit'
    #     primitives2[i]='movej_fit'

    ###ilc toolbox def
    ilc=ilc_toolbox([robot1,robot2],[primitives1,primitives2],base2_R,base2_p)

    ### save figure or show
    save_fig = True

    multi_peak_threshold=0.2
    ###TODO: extension fix start point, moveC support
    all_speed_profile1 = []
    all_speed_profile2 = []
    all_speed_std = []
    all_max_error = []

    iteration=55
    speed_iteration = 5
    draw_speed_max=None
    draw_error_max=None
    max_error_tolerance = 0.5
    speed_std_tolerance = 5
    max_error = 999
    max_ang_error = 999
    max_error_all_thres = 1 # threshold to push all bp
    # max_error_all_thres = 5 # threshold to push all bp
    alpha_break_thres=0.1

    start_iteration=0
    if use_exist:
        start_iteration=use_iteration
    for i in range(start_iteration,iteration):

        all_alpha_error=[]
        all_alpha=[]
        alpha_flag=False

        # get the best speed profile after speed update
        if i==speed_iteration+start_iteration:
            if start_iteration==0:
                std_id = np.squeeze(np.argwhere(np.array(all_speed_std)<speed_std_tolerance))
                print(std_id)
                print(np.array(all_max_error))
                min_err_id = np.squeeze(np.argwhere(np.array(all_max_error)==np.min(np.array(all_max_error)[std_id])))
                print(min_err_id)
                s1_movel = deepcopy(all_speed_profile1[min_err_id])
                s2_movel = deepcopy(all_speed_profile2[min_err_id])
                print(s1_movel)
                print(s2_movel)
        
        if i>speed_iteration+start_iteration:
            error_prev = deepcopy(error)
            if max(error) > max_error_all_thres:
                alpha = 1
            else:
                alpha = 1
        speed_alpha = 1

        while True:
            p_bp1_update = deepcopy(p_bp1)
            q_bp1_update = deepcopy(q_bp1)
            p_bp2_update = deepcopy(p_bp2)
            q_bp2_update = deepcopy(q_bp2)
            s1_movel_update = deepcopy(s1_movel)
            s2_movel_update = deepcopy(s2_movel)
            if i != start_iteration:
                
                # new robot data
                speed_exe1 = np.divide(np.linalg.norm(np.diff(curve_exe1,axis=0),2,1),np.diff(timestamp1,axis=0))
                speed_exe1=np.append(speed_exe1,speed_exe1[-1])
                speed_exe2 = np.divide(np.linalg.norm(np.diff(curve_exe2,axis=0),2,1),np.diff(timestamp2,axis=0))
                speed_exe2=np.append(speed_exe2,speed_exe2[-1])
                p_bp1_sq = np.squeeze(p_bp1[step_start1:step_end1+1])
                p_bp2_sq = np.squeeze(p_bp2[step_start2:step_end2+1])
                lam1_bp=np.cumsum(np.linalg.norm(np.diff(p_bp1_sq,axis=0),2,1))
                lam1_bp=np.append(0,lam1_bp)
                lam2_bp=np.cumsum(np.linalg.norm(np.diff(p_bp2_sq,axis=0),2,1))
                lam2_bp=np.append(0,lam2_bp)
                
                if i < speed_iteration+start_iteration:
                    ### update speed
                    # print(curve_exe1)
                    for j in range(step_start1+1,step_end1+1):
                        ###### update speed 1
                        error_temp,id_prev=calc_error(p_bp1[j-1][-1],curve_exe1)
                        error_temp,id_now=calc_error(p_bp1[j][-1],curve_exe1)
                        if id_prev<id_now+1:
                            this_speed_exe1 = np.mean(speed_exe1[id_prev:id_now+1])
                            # print(p_bp1[j][-1],id_prev,id_now,this_speed_exe1)
                            s1_movel[j] = (s1_movel_des[j]/this_speed_exe1)*s1_movel[j]
                            # s1_movel[j] = (s1_movel_des[j]-this_speed_exe1)+s1_movel[j]
                    # print(curve_exe2)
                    for j in range(step_start2+1,step_end2+1):
                        ###### update speed 2
                        error_temp,id_prev=calc_error(p_bp2[j-1][-1],curve_exe2)
                        error_temp,id_now=calc_error(p_bp2[j][-1],curve_exe2)
                        if id_prev<id_now+1:
                            this_speed_exe2 = np.mean(speed_exe2[id_prev:id_now+1])
                            # print(p_bp2[j][-1],id_prev,id_now,this_speed_exe2)
                            s2_movel[j] = (s2_movel_des[j]/this_speed_exe2)*s2_movel[j]
                            # s2_movel[j] = (s2_movel_des[j]-this_speed_exe2)+s2_movel[j]
                    s1_movel[step_end1+1:]=s1_movel[step_end1]
                    s2_movel[step_end2+1:]=s1_movel[step_end2]
                    s1_movel_update = deepcopy(s1_movel)
                    s2_movel_update = deepcopy(s2_movel)

                elif  i == speed_iteration+start_iteration:
                    pass # dont update at iteration "speed_iteration"
                else: # update bp
                    print(alpha)
                    if max(error_prev) > max_error_all_thres:
                        
                        # method 1: move point propotional to breakpoint distance, and only adjust R of arm 1
                        for j in range(0,len(p_bp1)):
                            if j==0:
                                arm1_d = np.linalg.norm(p_bp1[j+1][-1]-p_bp1[j][-1])
                                arm2_d = np.linalg.norm(p_bp2[j+1][-1]-p_bp2[j][-1])
                            else:
                                arm1_d = np.linalg.norm(p_bp1[j][-1]-p_bp1[j-1][-1])
                                arm2_d = np.linalg.norm(p_bp2[j][-1]-p_bp2[j-1][-1])
                            arm1_p = arm1_d/(arm1_d+arm2_d)
                            arm2_p = arm2_d/(arm1_d+arm2_d)
                            p_bp1_update[j][-1] = np.array(p_bp1[j][-1]) + alpha*arm1_p*error_dir1[j]
                            p_bp2_update[j][-1] = np.array(p_bp2[j][-1]) + alpha*arm2_p*error_dir2[j]
                            #### get the new R
                            bp1_R = robot1.fwd(q_bp1[j][-1]).R
                            bp1_R[:,2] = (bp1_R[:,2] + alpha*ang_error_dir1[j])/np.linalg.norm((bp1_R[:,2] + alpha*ang_error_dir1[j]))
                            bp1_R[:,0] = bp1_R[:,0]-np.dot(bp1_R[:,0],bp1_R[:,2])*bp1_R[:,2]
                            bp1_R[:,0] = bp1_R[:,0]/np.linalg.norm(bp1_R[:,0])
                            bp1_R[:,1] = np.cross(bp1_R[:,2],bp1_R[:,0])/np.linalg.norm(np.cross(bp1_R[:,2],bp1_R[:,0]))
                            ################
                            bp2_R = robot2.fwd(q_bp2[j][-1]).R
                            q_bp1_update[j][-1] = car2js(robot1, q_bp1[j][-1], p_bp1_update[j][-1], bp1_R)[0]
                            q_bp2_update[j][-1] = car2js(robot2, q_bp2[j][-1], p_bp2_update[j][-1], bp2_R)[0]
                        
                        # method 2: use qp to grow the original breakpoint to the new breakpoints
                        
                    else:
                        ### update only the max
                        print(len(error_prev))
                        print(peaks_dedp)
                        for j in range(len(peaks_dedp)):
                            peak = deepcopy(peaks_dedp[j])
                            de_dp= deepcopy(all_dedp[j])
                            p_bp1_update, q_bp1_update=ilc.update_bp_xyz(p_bp1,q_bp1,de_dp,error_prev[peak],breakpoint_interp_2tweak_indices,alpha=alpha)

                    #### update speed base on bp change
                    # p_bp1_sq = np.squeeze(p_bp1_update[step_start1:step_end1+1])
                    # dlam1_movel=np.linalg.norm(np.diff(p_bp1_sq,axis=0),2,1)
                    # update_ratio = np.divide(np.divide(dlam1_movel,dt_movel), s1_movel_des[step_start1+1:step_end1+1])
                    # s1_movel_update[step_start1+1:step_end1+1] = np.multiply(update_ratio,s1_movel[step_start1+1:step_end1+1])
                    # s1_movel_update[step_end1+1:]=s1_movel[step_end1]
                    ###################################

                    p_bp_relative_new,_=ms.form_relative_path(np.squeeze(q_bp1_update),np.squeeze(q_bp2),base2_R,base2_p)
                    ### update visualization
                    ax = plt.axes(projection='3d')
                    ax.plot3D(relative_path[:,0], relative_path[:,1],relative_path[:,2], 'red',label='original')
                    # ax.scatter3D(relative_path_exe[peaks,0], relative_path_exe[peaks,1], relative_path_exe[peaks,2],c='orange',label='worst case')
                    # ax.scatter3D(all_new_bp[:,0], all_new_bp[:,1], all_new_bp[:,2], c='magenta',label='new breakpoints')
                    ax.plot3D(relative_path_exe[:,0], relative_path_exe[:,1],relative_path_exe[:,2], 'green',label='execution')
                    ax.scatter3D(p_bp_relative[step_start1:step_end1+1,0], p_bp_relative[step_start1:step_end1+1,1],p_bp_relative[step_start1:step_end1+1,2], 'blue', label='old bps')
                    ax.scatter3D(p_bp_relative_new[step_start1:step_end1+1,0], p_bp_relative_new[step_start1:step_end1+1,1],p_bp_relative_new[step_start1:step_end1+1,2], 'magenta', label='new bps')
                    ax.view_init(61, -67)
                    plt.legend()
                    if save_fig:
                        plt.savefig(ilc_output+'traj_iteration_'+str(i-1))
                        plt.clf()
                    else:
                        plt.show()
                
                ## show speed of each robot
                if i<=start_iteration+1:
                    speed_exe1_old = np.zeros(len(curve_exe1))
                    speed_exe2_old = np.zeros(len(curve_exe2))
                    lam1_old = np.zeros(len(curve_exe1))
                    lam2_old = np.zeros(len(curve_exe2))
                f, ax = plt.subplots(1, 2)
                ax[0].scatter(lam1_bp,s1_movel_des[step_start1:step_end1+1],label='Motion Program',c='tab:green')
                ax[0].plot(lam1_old,speed_exe1_old,label='Prev Execution')
                ax[0].plot(lam1,speed_exe1,label='Execution')
                ax[1].scatter(lam2_bp,s2_movel_des[step_start2:step_end2+1],label='Motion Program',c='tab:green')
                ax[1].plot(lam2_old,speed_exe2_old,label='Prev Execution')
                ax[1].plot(lam2,speed_exe2,label='Execution')
                plt.legend()
                if save_fig:
                    plt.savefig(ilc_output+'speed_'+str(i-1))
                    plt.clf()
                else:
                    plt.show()
                speed_exe1_old = deepcopy(speed_exe1)
                speed_exe2_old = deepcopy(speed_exe2)
                lam1_old = deepcopy(lam1)
                lam2_old = deepcopy(lam2)
                ######

            ###execution with plant
            logged_data1,logged_data2=ms.exec_motions_multimove_separate2(robot1,robot2,primitives1,primitives2,p_bp1_update,p_bp2_update,q_bp1_update,q_bp2_update,s1_movel_update,s2_movel_update,z,z)
            StringData1=StringIO(logged_data1.decode('utf-8'))
            df1 = read_csv(StringData1, sep =",")
            StringData2=StringIO(logged_data2.decode('utf-8'))
            df2 = read_csv(StringData2, sep =",")

            ##############################data analysis#####################################
            lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R = ms.logged_data_analysis_multimove_connect(df1,df2,base2_R,base2_p,realrobot=False)
            #############################chop extension off##################################
            if i >= speed_iteration+start_iteration:
                lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
                    ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])
                timestamp1 = timestamp
                timestamp2 = timestamp
                lam1=np.append(0,np.cumsum(np.linalg.norm(np.diff(curve_exe1,axis=0),2,1)))
                lam2=np.append(0,np.cumsum(np.linalg.norm(np.diff(curve_exe2,axis=0),2,1)))
            else:
                # lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
                #     ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])
                curve_exe1_unchop = deepcopy(curve_exe1)
                curve_exe2_unchop = deepcopy(curve_exe2)
                curve_exe_R1_unchop = deepcopy(curve_exe_R1)
                curve_exe_R2_unchop = deepcopy(curve_exe_R2)
                speed_unchop = deepcopy(speed)
                timestamp_unchop = deepcopy(timestamp)
                lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
                        ms.chop_extension_dual_extend(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,[curve1[0],curve2[0]],[curve1[-1],curve2[-1]],[curve_exe1,curve_exe2])
                lam1, curve_exe1, curve_exe_R1,curve_exe_js1, speed1, timestamp1=ms.chop_extension(curve_exe1_unchop, curve_exe_R1_unchop,curve_exe_js1, speed_unchop, timestamp_unchop,curve1[:,:3],curve1[:,3:])
                lam2, curve_exe2, curve_exe_R2,curve_exe_js2, speed2, timestamp2=ms.chop_extension(curve_exe2_unchop, curve_exe_R2_unchop,curve_exe_js2, speed_unchop, timestamp_unchop,curve2[:,:3],curve2[:,3:])

            ##############################calcualte error########################################
            ave_speed=np.mean(speed)
            error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])

            # if error decrease
            print("Max error:",max(error),'Prev max error',max_error,max(np.degrees(angle_error)),max_ang_error)

            if i <= speed_iteration+start_iteration:
                all_speed_profile1.append(deepcopy(s1_movel))
                all_speed_profile2.append(deepcopy(s2_movel))
                all_speed_std.append(np.std(speed)/ave_speed*100)
                all_max_error.append(max(error))
                max_error=max(error)
                max_ang_error=max(np.degrees(angle_error))
                # if i==1:
                #     print(s1_movel)
                #     print(s2_movel)
                break
            
            # if max_error<max_error_tolerance:
            #     exit()

            ilc_output_sub=ilc_output+'iteration_'+str(i)+'/'
            Path(ilc_output_sub).mkdir(exist_ok=True)
            if draw_speed_max is None:
                draw_speed_max=max(speed)*1.05
            if max(speed) >= draw_speed_max or max(speed) < draw_speed_max*0.1:
                draw_speed_max=max(speed)*1.05
            if draw_error_max is None:
                draw_error_max=max(error)*1.05
            if max(error) >= draw_error_max or max(error) < draw_error_max*0.1:
                draw_error_max=max(error)*1.05
            #############################error peak detection###############################
            find_peak_dist = 20/(lam[int(len(lam)/2)]-lam[int(len(lam)/2)-1])
            if find_peak_dist<1:
                find_peak_dist=1
            peaks,_=find_peaks(error,height=multi_peak_threshold,prominence=0.05,distance=find_peak_dist)   ###only push down peaks higher than height, distance between each peak is 20mm, threshold to filter noisy peaks
            if len(peaks)==0 or np.argmax(error) not in peaks:
                peaks=np.append(peaks,np.argmax(error))
            
            ## save the results
            if not alpha_flag: # dont draw again is alpha flag raised
                error_speed_plot(lam,speed,error,peaks,angle_error,draw_speed_max,draw_error_max,ilc_output_sub,'alpha_'+str(round(alpha*100)),save_fig=True)
                df1.to_csv(ilc_output_sub+'alpha_'+str(round(alpha*100))+'_js1.csv',header=False,index=False)
                df2.to_csv(ilc_output_sub+'alpha_'+str(round(alpha*100))+'_js2.csv',header=False,index=False)
                dfcmd1=DataFrame({'primitives':primitives1,'points':p_bp1_update,'q_bp':q_bp1_update,'speed':s1_movel_update})
                dfcmd1.to_csv(ilc_output_sub+'alpha_'+str(round(alpha*100))+'command1.csv',header=True,index=False)
                dfcmd2=DataFrame({'primitives':primitives2,'points':p_bp2_update,'q_bp':q_bp2_update,'speed':s2_movel_update})
                dfcmd2.to_csv(ilc_output_sub+'alpha_'+str(round(alpha*100))+'command2.csv',header=True,index=False)

            # if max(error) < max_error and max(np.degrees(angle_error)) < max_ang_error:
            if (max(error) < max_error) or alpha_flag:
                print('max error break')
                max_error = max(error)
                max_ang_error = max(np.degrees(angle_error))
                p_bp1 = deepcopy(p_bp1_update)
                q_bp1 = deepcopy(q_bp1_update)
                s1_movel = deepcopy(s1_movel_update)
                p_bp2 = deepcopy(p_bp2_update)
                q_bp2 = deepcopy(q_bp2_update)
                s2_movel = deepcopy(s2_movel_update)
                break

            all_alpha_error.append(max(error))
            all_alpha.append(alpha)
            # if not, decrease alpha
            # alpha = alpha*0.9
            alpha = alpha*0.75
            # if too many iteration of alpha, still break
            if alpha < alpha_break_thres:
                print("alpha break")
                # choose the alpha with lowest max error
                print(all_alpha_error)
                print(all_alpha)
                alpha_min=np.argmin(all_alpha_error)
                print(alpha_min)
                alpha=all_alpha[alpha_min]
                print(alpha)
                alpha_flag=True

        ############################## 
        print('Iteration:',i,', Max Error:',max(error),'Ave. Speed:',ave_speed,'Std. Speed:',np.std(speed),'Std/Ave (%):',np.std(speed)/ave_speed*100)
        print('Max Speed:',max(speed),'Min Speed:',np.min(speed),'Ave. Error:',np.mean(error),'Min Error:',np.min(error),"Std. Error:",np.std(error))
        print('Max Ang Error:',max(np.degrees(angle_error)),'Min Ang Error:',np.min(np.degrees(angle_error)),'Ave. Ang Error:',np.mean(np.degrees(angle_error)),"Std. Ang Error:",np.std(np.degrees(angle_error)))
        print("===========================================")
        #############################error peak detection###############################
        find_peak_dist = 20/(lam[int(len(lam)/2)]-lam[int(len(lam)/2)-1])
        if find_peak_dist<1:
            find_peak_dist=1
        peaks,_=find_peaks(error,height=multi_peak_threshold,prominence=0.05,distance=find_peak_dist)   ###only push down peaks higher than height, distance between each peak is 20mm, threshold to filter noisy peaks
        if len(peaks)==0 or np.argmax(error) not in peaks:
            peaks=np.append(peaks,np.argmax(error))

        # peaks=np.array([np.argmax(error)])
        ##############################plot error#####################################
        if draw_speed_max is None:
            draw_speed_max=max(speed)*1.05
        if max(speed) >= draw_speed_max or max(speed) < draw_speed_max*0.1:
            draw_speed_max=max(speed)*1.05
        if draw_error_max is None:
            draw_error_max=max(error)*1.05
        if max(error) >= draw_error_max or max(error) < draw_error_max*0.1:
            draw_error_max=max(error)*1.05
        error_speed_plot(lam,speed,error,peaks,angle_error,draw_speed_max,draw_error_max,ilc_output,'iteration_'+str(i),save_fig=True)
        # exit()

        # save result
        df1.to_csv(ilc_output+'iter_'+str(i)+'_js1.csv',header=False,index=False)
        df2.to_csv(ilc_output+'iter_'+str(i)+'_js2.csv',header=False,index=False)
        df=DataFrame({'primitives':primitives1,'points':p_bp1,'q_bp':q_bp1,'speed':s1_movel})
        df.to_csv(ilc_output+'command_arm1_'+str(i)+'.csv',header=True,index=False)
        df=DataFrame({'primitives':primitives2,'points':p_bp2,'q_bp':q_bp2,'speed':s2_movel})
        df.to_csv(ilc_output+'command_arm2_'+str(i)+'.csv',header=True,index=False)

        if i >= speed_iteration:
            if max(error) > max_error_all_thres:
                ##########################################calculate error direction and push######################################
                ### interpolate curve (get gradient direction)
                curve_target = np.zeros((len(relative_path_exe), 3))
                curve_target_R = np.zeros((len(relative_path_exe), 3))
                for j in range(len(relative_path_exe)):
                    dist = np.linalg.norm(relative_path[:,:3] - relative_path_exe[j], axis=1)
                    closest_point_idx = np.argmin(dist)
                    curve_target[j, :] = relative_path[closest_point_idx, :3]
                    curve_target_R[j, :] = relative_path[closest_point_idx, 3:]

                ### get error (and transfer into robot1 frame)
                error_tool=[]
                angle_error_tool=[]
                error1 = []
                angle_error1 = []
                error2 = []
                for j in range(len(relative_path_exe)):
                    ## calculate error
                    error_t2 = curve_target[j]-relative_path_exe[j]
                    ang_error_t2 = curve_target_R[j]-relative_path_exe_R[j][:,-1]
                    error_tool.append(error_t2)
                    angle_error_tool.append(ang_error_t2)
                    ## calculate error direction in robot frame
                    R2 = robot2.fwd(curve_exe_js2[j]).R
                    R1 = np.matmul(base2_R,R2)
                    error1.append(np.dot(R1,error_t2))
                    error2.append(np.dot(R2,error_t2))
                    angle_error1.append(np.dot(R1,ang_error_t2))
                ### get closets bp index
                p_bp_relative,_=ms.form_relative_path(np.squeeze(q_bp1),np.squeeze(q_bp2),base2_R,base2_p)
                # find error direction
                error_dir=[]
                angle_error_dir=[]
                error_dir1=[]
                ang_error_dir1=[]
                error_dir2=[]
                for j in range(0, len(p_bp_relative)): # include first and the last bp and the bp for extension
                    p_bp = deepcopy(p_bp_relative[j])
                    closest_point_idx = np.argmin(np.linalg.norm(curve_target - p_bp, axis=1))
                    error_dir.append(deepcopy(error_tool[closest_point_idx]))
                    angle_error_dir.append(deepcopy(angle_error_tool[closest_point_idx]))
                    error_dir1.append(deepcopy(error1[closest_point_idx]))
                    error_dir2.append(deepcopy(error2[closest_point_idx]))
                    ang_error_dir1.append(deepcopy(angle_error1[closest_point_idx]))
                
            else:
                all_dedp = []
                ##########################################calculate gradient for peaks######################################
                ###restore trajectory from primitives
                curve_interp1, curve_R_interp1, curve_js_interp1, breakpoints_blended=form_traj_from_bp(q_bp1,primitives1,robot1)
                curve_interp2, curve_R_interp2, curve_js_interp2, breakpoints_blended=form_traj_from_bp(q_bp2,primitives2,robot2)
                curve_js_blended1,curve_blended1,curve_R_blended1=blend_js_from_primitive(curve_interp1, curve_js_interp1, breakpoints_blended, primitives1,robot1,zone=10)
                curve_js_blended2,curve_blended2,curve_R_blended2=blend_js_from_primitive(curve_interp2, curve_js_interp2, breakpoints_blended, primitives2,robot2,zone=10)

                ###establish relative trajectory from blended trajectory
                relative_path_blended,relative_path_blended_R=ms.form_relative_path(curve_js_blended1,curve_js_blended2,base2_R,base2_p)

                # all_new_bp=[]
                peaks_dedp = deepcopy(peaks)
                for peak in peaks_dedp:
                    ######gradient calculation related to nearest 3 points from primitive blended trajectory, not actual one
                    _,peak_error_curve_idx=calc_error(relative_path_exe[peak],relative_path[:,:3])  # index of original curve closest to max error point

                    ###get closest to worst case point on blended trajectory
                    _,peak_error_curve_blended_idx=calc_error(relative_path_exe[peak],relative_path_blended)

                    ###############get numerical gradient#####
                    ###find closest 3 breakpoints
                    order=np.argsort(np.abs(breakpoints_blended-peak_error_curve_blended_idx))
                    breakpoint_interp_2tweak_indices=order[:3]
                    print("breakpoints to twist:",breakpoint_interp_2tweak_indices)

                    # calculate desired robot1 point
                    tool_in_base1 = rox.Transform(base2_R,base2_p)*robot2.fwd(curve_exe_js2[peak])
                    closest_T = tool_in_base1*rox.Transform(np.eye(3),relative_path[peak_error_curve_idx,:3])
                    closest_p=closest_T.p
                    
                    de_dp=ilc.get_gradient_from_model_xyz_fanuc(p_bp1,q_bp1,breakpoints_blended,curve_blended1,peak_error_curve_blended_idx,robot1.fwd(curve_exe_js1[peak]),closest_p,breakpoint_interp_2tweak_indices,ave_speed)
                    all_dedp.append(de_dp)


if __name__ == "__main__":
    main()