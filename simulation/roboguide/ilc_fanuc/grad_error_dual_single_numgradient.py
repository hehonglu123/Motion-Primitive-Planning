########
# This module utilized https://github.com/eric565648/fanuc_motion_program_exec_client
# and send whatever the motion primitives that algorithms generate
# to Roboguide
########

from copy import deepcopy
import numpy as np
from scipy.interpolate import interp1d
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

def main():
    # curve
    # data_type='blade'
    # data_type='wood'
    # data_type='blade_arm_shift'
    data_type='blade_base_shift'

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

    test_type='dual_arm'
    # test_type='dual_single_arm'
    # test_type='dual_single_arm_straight' # robot2 is multiple user defined straight line
    # test_type='dual_single_arm_straight_50' # robot2 is multiple user defined straight line
    # test_type='dual_single_arm_straight_min' # robot2 is multiple user defined straight line
    # test_type='dual_single_arm_straight_min10' # robot2 is multiple user defined straight line
    # test_type='dual_arm_10'
    # test_type='dual_arm_qp'

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
        ms = MotionSendFANUC(robot1=robot1,robot2=robot2)
    elif data_type=='wood':
        ms = MotionSendFANUC(robot1=robot1,robot2=robot2,utool2=3)
    elif data_type=='blade_arm_shift':
        ms = MotionSendFANUC(robot1=robot1,robot2=robot2,utool2=6)
    elif data_type=='blade_base_shift':
        ms = MotionSendFANUC(robot1=robot1,robot2=robot2,utool2=2)

    # s=int(1600/2.) # mm/sec in leader frame
    s=1000 # mm/sec
    # s=16 # mm/sec in leader frame
    z=100 # CNT100
    ilc_output=data_dir+'results_'+str(s)+'_'+test_type+'/'
    Path(ilc_output).mkdir(exist_ok=True)

    breakpoints1,primitives1,p_bp1,q_bp1,_=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command1.csv')
    breakpoints2,primitives2,p_bp2,q_bp2,_=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command2.csv')

    # for i in range(len(q_bp2)):
    #     p_bp2[i][-1] = robot2.fwd(q_bp2[i][-1]).p

    ### dlam analysis
    # fig, ax = plt.subplots(1, 2)
    # dlam1=np.linalg.norm(np.diff(curve1,axis=0),2,1)
    # ax[0].plot(dlam1)
    # dlam2=np.linalg.norm(np.diff(curve2,axis=0),2,1)
    # ax[1].plot(dlam2)
    # plt.title('dlam')
    # plt.show()
    # ### dlam movel analysis
    # fig, ax = plt.subplots(1, 3)
    # p_bp1_sq = np.squeeze(p_bp1)
    # p_bp2_sq = np.squeeze(p_bp2)
    # dlam1=np.linalg.norm(np.diff(p_bp1_sq,axis=0),2,1)
    # ax[0].plot(dlam1)
    # dlam2=np.linalg.norm(np.diff(p_bp2_sq,axis=0),2,1)
    # ax[1].plot(dlam2)
    # ### dlam movel relative
    # p_bp_relative,_=ms.form_relative_path(np.squeeze(q_bp1),np.squeeze(q_bp2),base2_R,base2_p)
    # dlam=np.linalg.norm(np.diff(p_bp_relative,axis=0),2,1)
    # ax[2].plot(dlam)
    # plt.title('movel dlam')
    # plt.show()

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

    # use_exist=False
    # use_iteration=231
    use_exist=True
    use_iteration=27

    read_cmd_folder = data_dir+'results_1000_dual_arm_extend1_nospeedreg/'
    if use_exist:
        _,primitives1,p_bp1,q_bp1,s1_movel=ms.extract_data_from_cmd(os.getcwd()+'/'+read_cmd_folder+'command_arm1_'+str(use_iteration)+'.csv')
        _,primitives2,p_bp2,q_bp2,s2_movel=ms.extract_data_from_cmd(os.getcwd()+'/'+read_cmd_folder+'command_arm2_'+str(use_iteration)+'.csv')
    
    for i in range(len(p_bp1)):
        p_bp1[i][-1]=robot1.fwd(q_bp1[i][-1]).p
    for i in range(len(p_bp2)):
        p_bp2[i][-1]=robot2.fwd(q_bp2[i][-1]).p

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

    iteration=40
    speed_iteration = 0
    draw_speed_max=None
    draw_error_max=None
    max_error_tolerance = 0.5
    speed_std_tolerance = 5
    max_error = 999
    max_ang_error = 999
    max_error_all_thres = 1 # threshold to push all bp
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
            if len(all_speed_profile1) != 0:
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
                
                Ks = 10
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
                        ### update all p_bp1
                        for j in range(step_start1,step_end1+1):
                            p_bp1_update[j][-1] = np.array(p_bp1[j][-1]) + alpha*p_bp1_error_dir[j-step_start1]
                            bp1_R = robot1.fwd(q_bp1[j][-1]).R
                            bp1_R[:,-1] = (bp1_R[:,-1] + alpha*p_bp1_ang_error_dir[j-step_start1])/np.linalg.norm((bp1_R[:,-1] + alpha*p_bp1_ang_error_dir[j-step_start1]))
                            q_bp1_update[j][-1] = car2js(robot1, q_bp1[j][-1], p_bp1_update[j][-1], bp1_R)[0]
                    else:
                        ### update only the max
                        ### already update
                        alpha_flag=True
                        pass

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
            logged_data=ms.exec_motions_multimove_nocoord(robot1,robot2,primitives1,primitives2,p_bp1_update,p_bp2,q_bp1_update,q_bp2,s1_movel_update,s2_movel_update,z,z)
            # with open('iteration_'+str(i)+'.csv',"wb") as f:
            #     f.write(logged_data)
            StringData=StringIO(logged_data.decode('utf-8'))
            df = read_csv(StringData, sep =",")
            ##############################data analysis#####################################
            lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R = ms.logged_data_analysis_multimove(df,base2_R,base2_p,realrobot=False)
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
            print(max(error),max_error,max(np.degrees(angle_error)),max_ang_error)

            if i <= speed_iteration+start_iteration:
                all_speed_profile1.append(deepcopy(s1_movel))
                all_speed_profile2.append(deepcopy(s2_movel))
                all_speed_std.append(np.std(speed)/ave_speed*100)
                all_max_error.append(max(error))
                max_error=max(error)
                break
            
            # if max(error) < max_error and max(np.degrees(angle_error)) < max_ang_error:
            if (max(error) < max_error) or alpha_flag:
                print('max error break')
                max_error = max(error)
                max_ang_error = max(np.degrees(angle_error))
                p_bp1 = deepcopy(p_bp1_update)
                q_bp1 = deepcopy(q_bp1_update)
                s1_movel = deepcopy(s1_movel_update)
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
        peaks,_=find_peaks(error,height=multi_peak_threshold,prominence=0.05,distance=find_peak_dist)		###only push down peaks higher than height, distance between each peak is 20mm, threshold to filter noisy peaks
        if len(peaks)==0 or np.argmax(error) not in peaks:
            peaks=np.append(peaks,np.argmax(error))

        # peaks=np.array([np.argmax(error)])
        ##############################plot error#####################################
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(lam, speed, 'g-', label='Speed')
        ax2.plot(lam, error, 'b-',label='Error')
        ax2.scatter(lam[peaks],error[peaks],label='peaks')
        ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
        if draw_speed_max is None:
            draw_speed_max=max(speed)*1.05
        if max(speed) >= draw_speed_max or max(speed) < draw_speed_max*0.1:
            draw_speed_max=max(speed)*1.05
        ax1.axis(ymin=0,ymax=draw_speed_max)
        if draw_error_max is None:
            draw_error_max=max(error)*1.05
        if max(error) >= draw_error_max or max(error) < draw_error_max*0.1:
            draw_error_max=max(error)*1.05
        ax2.axis(ymin=0,ymax=draw_error_max)
        ax1.set_xlabel('lambda (mm)')
        ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
        ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
        plt.title("Speed and Error Plot")
        ax1.legend(loc=0)
        ax2.legend(loc=0)
        plt.legend()
        if save_fig:
            plt.savefig(ilc_output+'iteration_'+str(i))
            plt.clf()
        else:
            plt.show()
        # exit()

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
                curve_target1 = []
                curve_target_R1 = []
                error1 = []
                angle_error1 = []
                for j in range(len(relative_path_exe)):
                    ## transfer to robot1 frame
                    tool_in_base1 = rox.Transform(base2_R,base2_p)*robot2.fwd(curve_exe_js2[j])
                    this_curve_target1_T = tool_in_base1*rox.Transform(np.eye(3),curve_target[j])
                    this_curve_target_R1 = np.matmul(tool_in_base1.R,curve_target_R[j])
                    curve_target1.append(this_curve_target1_T.p)
                    curve_target_R1.append(this_curve_target_R1)
                    ## calculate error
                    error1.append(this_curve_target1_T.p-curve_exe1[j])
                    # angle_error1.append(get_angle(curve_exe_R1[j][:,-1], this_curve_target_R1))
                    angle_error1.append(this_curve_target_R1-curve_exe_R1[j][:,-1])

                ### get closets bp index
                p_bp_relative,_=ms.form_relative_path(np.squeeze(q_bp1),np.squeeze(q_bp2),base2_R,base2_p)
                p_bp1_error_dir=[]
                p_bp1_ang_error_dir=[]
                # find error direction
                for j in range(step_start1, step_end1+1): # exclude first and the last bp and the bp for extension
                    p_bp = p_bp_relative[j]
                    closest_point_idx = np.argmin(np.linalg.norm(curve_target - p_bp, axis=1))
                    error_dir = error1[closest_point_idx]
                    p_bp1_error_dir.append(error_dir)
                    ang_error_dir = angle_error1[closest_point_idx]
                    p_bp1_ang_error_dir.append(ang_error_dir)
                # p_bp1_error_dir.append(np.array([0,0,0])) # no update on the end breakpoints
                # p_bp1_error_dir.insert(0,np.array([0,0,0])) # no update on the first breakpoints
                # p_bp1_ang_error_dir.append(np.array([0,0,0])) # no update on the end breakpoints
                # p_bp1_ang_error_dir.insert(0,np.array([0,0,0])) # no update on the first breakpoints
            else:
                # variables for calculate gradient
                epsilon = 0.5
                backward_range = -19
                forward_range = 21
                ##########################################calculate gradient for peaks######################################
                # for peak in peaks:
                # only calculate the max peak
                # becaue adjust bp defintely affect others
                # also the duration is similar
                max_peaks=[np.argmax(error)] 
                for peak in max_peaks:
                    p_bp1_temp = deepcopy(p_bp1)
                    q_bp1_temp = deepcopy(q_bp1)
                    
                    ### interpolate curve (get gradient direction)
                    curve_target = np.zeros((len(relative_path_exe), 3))
                    curve_target_R = np.zeros((len(relative_path_exe), 3))
                    for j in range(len(relative_path_exe)):
                        dist = np.linalg.norm(relative_path[:,:3] - relative_path_exe[j], axis=1)
                        closest_point_idx = np.argmin(dist)
                        curve_target[j, :] = relative_path[closest_point_idx, :3]
                        curve_target_R[j, :] = relative_path[closest_point_idx, 3:]
                    ### get error (and transfer into robot1 frame)
                    curve_target1 = []
                    curve_target_R1 = []
                    error1 = []
                    angle_error1 = []
                    for j in range(len(relative_path_exe)):
                        ## transfer to robot1 frame
                        tool_in_base1 = rox.Transform(base2_R,base2_p)*robot2.fwd(curve_exe_js2[j])
                        this_curve_target1_T = tool_in_base1*rox.Transform(np.eye(3),curve_target[j])
                        this_curve_target_R1 = np.matmul(tool_in_base1.R,curve_target_R[j])
                        curve_target1.append(this_curve_target1_T.p)
                        curve_target_R1.append(this_curve_target_R1)
                        ## calculate error
                        error1.append(this_curve_target1_T.p-curve_exe1[j])
                        # angle_error1.append(get_angle(curve_exe_R1[j][:,-1], this_curve_target_R1))
                        angle_error1.append(this_curve_target_R1-curve_exe_R1[j][:,-1])
                    
                    p_bp_relative,_=ms.form_relative_path(np.squeeze(q_bp1_temp),np.squeeze(q_bp1_temp),base2_R,base2_p)
                    # get closets bp
                    order_id = np.argsort(np.linalg.norm(p_bp_relative-relative_path_exe[peak],2,1))
                    closest_bp_id=order_id[0]
                    # get timestamp range
                    prev_start = peak+backward_range if peak+backward_range>=0 else 0
                    prev_end = peak+forward_range if peak+forward_range<=len(timestamp1) else len(timestamp1)
                    timestamp_prev = deepcopy(timestamp1[prev_start:prev_end])
                    error1_dir_prev = deepcopy(error1[prev_start:prev_end])
                    error1_prev = np.linalg.norm(error1_dir_prev,2,1)

                    # iterate to get gradient
                    timestamp_xyz = []
                    curve_xyz_dp=[]
                    # adjust xyz
                    for pos_i in range(3):
                        p_bp1_temp = deepcopy(p_bp1)
                        q_bp1_temp=deepcopy(q_bp1)
                        p_bp1_temp[closest_bp_id][-1][pos_i] += epsilon
                        q_bp1_temp[closest_bp_id][-1]=car2js(robot1,q_bp1[closest_bp_id][-1],np.array(p_bp1_temp[closest_bp_id][-1]),robot1.fwd(q_bp1[closest_bp_id][-1]).R)[0]

                        logged_data=ms.exec_motions_multimove_nocoord(robot1,robot2,primitives1,primitives2,p_bp1_temp,p_bp2,q_bp1_temp,q_bp2,s1_movel,s2_movel,z,z)
                        StringData=StringIO(logged_data.decode('utf-8'))
                        df = read_csv(StringData, sep =",")
                        ##############################data analysis#####################################
                        _, curve_exe1_forgrad,_,_,_,_,_,_,timestamp_forgrad,_,_ = ms.logged_data_analysis_multimove(df,base2_R,base2_p,realrobot=False)
                        curve_exe1_forgrad=np.array(curve_exe1_forgrad)
                        this_timestamp=[]
                        this_curve_dx=[]
                        this_curve_dy=[]
                        this_curve_dz=[]
                        # d of the whole traj
                        for ti in range(len(timestamp1)):
                            t=timestamp1[ti]
                            if t not in timestamp_forgrad:
                                continue
                            curve_i = np.argwhere(timestamp_forgrad==t)[0][0]
                            this_timestamp.append(t)
                            this_curve_dx.append(curve_exe1_forgrad[curve_i][0]-curve_exe1[ti][0])
                            this_curve_dy.append(curve_exe1_forgrad[curve_i][1]-curve_exe1[ti][1])
                            this_curve_dz.append(curve_exe1_forgrad[curve_i][2]-curve_exe1[ti][2])

                        # print(this_timestamp)
                        # print(timestamp_prev)
                        dx_interp = interp1d(this_timestamp,this_curve_dx,kind='cubic',fill_value="extrapolate")(timestamp_prev)
                        dy_interp = interp1d(this_timestamp,this_curve_dy,kind='cubic',fill_value="extrapolate")(timestamp_prev)
                        dz_interp = interp1d(this_timestamp,this_curve_dz,kind='cubic',fill_value="extrapolate")(timestamp_prev)
                        this_curve_dp=[dx_interp,dy_interp,dz_interp]
                        this_curve_dp=np.reshape(np.array(this_curve_dp).T,(-1,))

                        timestamp_xyz.append(np.array(timestamp_prev))
                        curve_xyz_dp.append(this_curve_dp)

                        marker_size=2
                        fig, ax = plt.subplots(3,1)
                        # ax[0].plot(timestamp_forgrad,curve_exe1_forgrad[:,0],'-bo',markersize=marker_size) # x deviation
                        # ax[0].plot(timestamp_prev,curve_exe1_forgrad_x,'-go',markersize=marker_size)
                        # ax[0].set_title('traj new, x deviation')
                        # ax[1].plot(timestamp_forgrad,curve_exe1_forgrad[:,1],'-bo',markersize=marker_size) # y deviation
                        # ax[1].plot(timestamp_prev,curve_exe1_forgrad_y,'-go',markersize=marker_size)
                        # ax[1].set_title('traj new, y deviation')
                        # ax[2].plot(timestamp_forgrad,curve_exe1_forgrad[:,2],'-bo',markersize=marker_size) # z deviation
                        # ax[2].plot(timestamp_prev,curve_exe1_forgrad_z,'-go',markersize=marker_size)
                        # ax[2].set_title('traj new, z deviation')
                        # plt.show()
                        # ax[0].plot(this_timestamp,this_curve_dx,'-bo',markersize=marker_size) # x deviation
                        # ax[0].plot(timestamp_prev,dx_interp,'-go',markersize=marker_size)
                        # ax[0].set_title('traj new, x deviation')
                        # ax[1].plot(this_timestamp,this_curve_dy,'-bo',markersize=marker_size) # y deviation
                        # ax[1].plot(timestamp_prev,dy_interp,'-go',markersize=marker_size)
                        # ax[1].set_title('traj new, y deviation')
                        # ax[2].plot(this_timestamp,this_curve_dz,'-bo',markersize=marker_size) # z deviation
                        # ax[2].plot(timestamp_prev,dz_interp,'-go',markersize=marker_size)
                        # ax[2].set_title('traj new, z deviation')
                        # plt.show()

                    G = np.array(curve_xyz_dp).T*(1./epsilon)
                    
                    # decent loop
                    alpha = 0.5
                    while True:
                        du = alpha*np.matmul(np.linalg.pinv(G),np.reshape(np.array(error1_dir_prev),(-1,)))
                        p_bp1_temp = deepcopy(p_bp1)
                        q_bp1_temp=deepcopy(q_bp1)
                        p_bp1_temp[closest_bp_id][-1] = np.array(p_bp1_temp[closest_bp_id][-1])+du
                        q_bp1_temp[closest_bp_id][-1]=car2js(robot1,q_bp1[closest_bp_id][-1],np.array(p_bp1_temp[closest_bp_id][-1]),robot1.fwd(q_bp1[closest_bp_id][-1]).R)[0]
                        ###execution with plant
                        logged_data=ms.exec_motions_multimove_nocoord(robot1,robot2,primitives1,primitives2,p_bp1_temp,p_bp2,q_bp1_temp,q_bp2,s1_movel,s2_movel,z,z)
                        df = read_csv(StringIO(logged_data.decode('utf-8')), sep =",")
                        ##############################data analysis#####################################
                        lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R = ms.logged_data_analysis_multimove(df,base2_R,base2_p,realrobot=False)
                        #############################chop extension off##################################
                        lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
                            ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])
                        timestamp1 = timestamp
                        timestamp2 = timestamp
                        lam1=np.append(0,np.cumsum(np.linalg.norm(np.diff(curve_exe1,axis=0),2,1)))
                        lam2=np.append(0,np.cumsum(np.linalg.norm(np.diff(curve_exe2,axis=0),2,1)))
                        #############################################################################
                        error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])
                        if max(error) < max(error1_prev):
                            print("find alpha and du")
                            break
                        # if error not decrease
                        alpha = 0.75*alpha
                        if alpha<0.05:
                            print("Very small alpha")
                            break
                    p_bp1=deepcopy(p_bp1_temp)
                    q_bp1=deepcopy(q_bp1_temp)

                # exit()


if __name__ == "__main__":
    main()