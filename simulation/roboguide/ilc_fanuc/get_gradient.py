
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

    # for i in range(len(primitives1)):
    #     primitives1[i]='movej_fit'
    #     primitives2[i]='movej_fit'

    ###ilc toolbox def
    ilc=ilc_toolbox([robot1,robot2],[primitives1,primitives2],base2_R,base2_p)

    ### save figure or show
    save_fig = False
    save_data = False

    multi_peak_threshold=0.2
    ###TODO: extension fix start point, moveC support
    all_speed_profile1 = []
    all_speed_profile2 = []
    all_speed_std = []
    all_max_error = []

    
    speed_iteration = 2
    iteration=speed_iteration
    draw_speed_max=None
    draw_error_max=None
    max_error_tolerance = 0.5
    speed_std_tolerance = 5
    max_error = 999
    max_ang_error = 999
    max_error_all_thres = 1 # threshold to push all bp
    alpha_break_thres=0.1

    ########## read data ##############
    use_iteration=27
    cmd_folder = data_dir+'results_1000_dual_arm_extend1_nospeedreg/'
    _,primitives1,p_bp1,q_bp1,s1_movel_update=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_folder+'command_arm1_'+str(use_iteration)+'.csv')
    _,primitives2,p_bp2,q_bp2,s2_movel_update=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_folder+'command_arm2_'+str(use_iteration)+'.csv')

    _,_,_,_,s1_movel_update=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_folder+'command_arm1_'+str(27)+'.csv')
    _,_,_,_,s2_movel_update=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_folder+'command_arm2_'+str(27)+'.csv')

    # df=DataFrame({'primitives':primitives1,'points':p_bp1,'q_bp':q_bp1,'speed':s1_movel_update})
    # df.to_csv(os.getcwd()+'/'+cmd_folder+'command_arm1_'+str(use_iteration)+'.csv',header=True,index=False)
    # df=DataFrame({'primitives':primitives2,'points':p_bp2,'q_bp':q_bp2,'speed':s2_movel_update})
    # df.to_csv(os.getcwd()+'/'+cmd_folder+'command_arm2_'+str(use_iteration)+'.csv',header=True,index=False)
    # exit()

    for i in range(len(p_bp1)):
        p_bp1[i][-1]=robot1.fwd(q_bp1[i][-1]).p
    for i in range(len(p_bp2)):
        p_bp2[i][-1]=robot2.fwd(q_bp2[i][-1]).p

    ###execution with plant
    logged_data=ms.exec_motions_multimove_nocoord(robot1,robot2,primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,s1_movel_update,s2_movel_update,z,z)
    # with open('iteration_'+str(i)+'.csv',"wb") as f:
    #     f.write(logged_data)
    StringData=StringIO(logged_data.decode('utf-8'))
    df = read_csv(StringData, sep =",")
    ##############################data analysis#####################################
    lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R = ms.logged_data_analysis_multimove(df,base2_R,base2_p,realrobot=False)
    #############################chop extension off##################################
    lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
        ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])
    timestamp1 = timestamp
    timestamp2 = timestamp
    lam1=np.append(0,np.cumsum(np.linalg.norm(np.diff(curve_exe1,axis=0),2,1)))
    lam2=np.append(0,np.cumsum(np.linalg.norm(np.diff(curve_exe2,axis=0),2,1)))
    error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])
    
    #############################error peak detection###############################
    find_peak_dist = 20/(lam[int(len(lam)/2)]-lam[int(len(lam)/2)-1])
    if find_peak_dist<1:
        find_peak_dist=1
    peaks,_=find_peaks(error,height=multi_peak_threshold,prominence=0.05,distance=find_peak_dist)		###only push down peaks higher than height, distance between each peak is 20mm, threshold to filter noisy peaks
    if len(peaks)==0 or np.argmax(error) not in peaks:
        peaks=np.append(peaks,np.argmax(error))

    curve_exe_js1=np.array(curve_exe_js1)
    curve_exe_js2=np.array(curve_exe_js2)

    if save_data:
        # save js
        df1=DataFrame({'q0':curve_exe_js1[:,0],'q1':curve_exe_js1[:,1],'q2':curve_exe_js1[:,2],'q3':curve_exe_js1[:,3],'q4':curve_exe_js1[:,4],'q5':curve_exe_js1[:,5]})
        df1.to_csv(cmd_folder+'iter_'+str(use_iteration)+'_curve_exe_js1.csv',header=False,index=False)
        df2=DataFrame({'q0':curve_exe_js2[:,0],'q1':curve_exe_js2[:,1],'q2':curve_exe_js2[:,2],'q3':curve_exe_js2[:,3],'q4':curve_exe_js2[:,4],'q5':curve_exe_js2[:,5]})
        df2.to_csv(cmd_folder+'iter_'+str(use_iteration)+'_curve_exe_js2.csv',header=False,index=False)
        dft=DataFrame({'t':timestamp})
        dft.to_csv(cmd_folder+'iter_'+str(use_iteration)+'_timestamp.csv',header=False,index=False)

    ##############################plot error#####################################
    show_pics=False
    if show_pics:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(lam, speed, 'g-', label='Speed')
        ax2.plot(lam, error, '-bo', markersize=2,label='Error')
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

    print(peaks)
    # for the_peak in peaks[-1:]:
    #     print(the_peak)
    #     ########## calculate gradient with analytical model #######
    #     ### calculate gradient for one peak ###
    #     ###restore trajectory from primitives
    #     curve_interp1, curve_R_interp1, curve_js_interp1, breakpoints_blended=form_traj_from_bp(q_bp1,primitives1,robot1)
    #     curve_interp2, curve_R_interp2, curve_js_interp2, breakpoints_blended=form_traj_from_bp(q_bp2,primitives2,robot2)
    #     curve_js_blended1,curve_blended1,curve_R_blended1=blend_js_from_primitive(curve_interp1, curve_js_interp1, breakpoints_blended, primitives1,robot1,zone=10)
    #     curve_js_blended2,curve_blended2,curve_R_blended2=blend_js_from_primitive(curve_interp2, curve_js_interp2, breakpoints_blended, primitives2,robot2,zone=10)

    #     ###establish relative trajectory from blended trajectory
    #     relative_path_blended,relative_path_blended_R=ms.form_relative_path(curve_js_blended1,curve_js_blended2,base2_R,base2_p)

    #     ######gradient calculation related to nearest 3 points from primitive blended trajectory, not actual one
    #     _,peak_error_curve_idx=calc_error(relative_path_exe[the_peak],relative_path[:,:3])  # index of original curve closest to max error point

    #     ###get closest to worst case point on blended trajectory
    #     _,peak_error_curve_blended_idx=calc_error(relative_path_exe[the_peak],relative_path_blended)

    #     ###############get numerical gradient#####
    #     ###find closest 3 breakpoints
    #     order=np.argsort(np.abs(breakpoints_blended-peak_error_curve_blended_idx))
    #     breakpoint_interp_2tweak_indices=order[:3]

    #     # calculate desired robot1 point
    #     tool_in_base1 = rox.Transform(base2_R,base2_p)*robot2.fwd(curve_exe_js2[the_peak])
    #     closest_T = tool_in_base1*rox.Transform(np.eye(3),relative_path[peak_error_curve_idx,:3])
    #     closest_p=closest_T.p
        
    #     print(breakpoint_interp_2tweak_indices)
    #     de_dp=ilc.get_gradient_from_model_xyz_fanuc(p_bp1,q_bp1,breakpoints_blended,curve_blended1,peak_error_curve_blended_idx,robot1.fwd(curve_exe_js1[the_peak]),closest_p,breakpoint_interp_2tweak_indices,None)
        
    #     print(de_dp)
    #     print(np.linalg.pinv(de_dp))
    #     error_dir=robot1.fwd(curve_exe_js1[the_peak]).p-closest_p
    #     print(error_dir)
    #     print(error[the_peak])
    
    # p_bp1_update, q_bp1_update=ilc.update_bp_xyz(p_bp1,q_bp1,de_dp,error[the_peak],breakpoint_interp_2tweak_indices,alpha=1)
    # print(robot1.fwd(q_bp1[breakpoint_interp_2tweak_indices[0]][-1]).p)
    # print(robot1.fwd(q_bp1[breakpoint_interp_2tweak_indices[1]][-1]).p)
    # print(robot1.fwd(q_bp1[breakpoint_interp_2tweak_indices[2]][-1]).p)
    # print(robot1.fwd(q_bp1_update[breakpoint_interp_2tweak_indices[0]][-1]).p)
    # print(robot1.fwd(q_bp1_update[breakpoint_interp_2tweak_indices[1]][-1]).p)
    # print(robot1.fwd(q_bp1_update[breakpoint_interp_2tweak_indices[2]][-1]).p)
    
    # exit()

    ########## calculate numerical gradient here #############
    ## variables
    epsilon = 0.5
    backward_range = -19
    forward_range = 21
    the_peak = peaks[2] # for iteration 27
    # change of bp v.s. change in trajectory

    # get closest bp
    order_id = np.argsort(np.linalg.norm(p_bp_relative-relative_path_exe[the_peak],2,1))
    closest_bp_id=order_id[0]
    print(closest_bp_id)
    second_closest_bp_id=order_id[1]
    third_closest_bp_id = closest_bp_id+1 if closest_bp_id > second_closest_bp_id else closest_bp_id-1

    # ax = plt.axes(projection='3d')
    # # ax.plot3D(relative_path[:,0], relative_path[:,1],relative_path[:,2], 'red',label='original')
    # ax.plot3D(relative_path_exe[:,0], relative_path_exe[:,1],relative_path_exe[:,2], '-go',label='execution')
    # ax.scatter3D(p_bp_relative[:,0], p_bp_relative[:,1],p_bp_relative[:,2], 'blue', label='bps')
    # ax.view_init(61, -67)
    # plt.legend()
    # plt.show()

    # get closest exe
    # closest_exe_id = np.argmin(np.linalg.norm(relative_path_exe-p_bp_relative[closest_bp_id][-1],2,1))
    # # _,closest_exe_id = calc_error(p_bp_relative[closest_bp_id][-1],relative_path_exe)
    # print(closest_exe_id)
    # exit()

    ## curve 27
    # the_peak=closest_exe_id
    prev_start = the_peak+backward_range if the_peak+backward_range>=0 else 0
    prev_end = the_peak+forward_range if the_peak+forward_range<=len(timestamp1) else len(timestamp1)
    timestamp_prev = deepcopy(timestamp1[prev_start:prev_end])
    peak_time = timestamp1[the_peak]
    curve_prev = deepcopy(curve_exe1[prev_start:prev_end])
    peak_curve = curve_exe1[the_peak]
    p_bp_relative,_=ms.form_relative_path(np.squeeze(q_bp1),np.squeeze(q_bp2),base2_R,base2_p)
    print(timestamp_prev)

    timestamp_xyz = []
    curve_xyz_dx = []
    curve_xyz_dy = []
    curve_xyz_dz = []
    # adjust xyz
    for pos_i in range(3):
        p_bp1_temp = deepcopy(p_bp1)
        q_bp1_temp=np.array(deepcopy(q_bp1))
        p_bp1_temp[closest_bp_id][-1][pos_i] += epsilon
        q_bp1_temp[closest_bp_id][-1]=car2js(robot1,q_bp1[closest_bp_id][-1],np.array(p_bp1_temp[closest_bp_id][-1]),robot1.fwd(q_bp1[closest_bp_id][-1]).R)[0]

        logged_data=ms.exec_motions_multimove_nocoord(robot1,robot2,primitives1,primitives2,p_bp1_temp,p_bp2,q_bp1_temp,q_bp2,s1_movel_update,s2_movel_update,z,z)
        StringData=StringIO(logged_data.decode('utf-8'))
        df = read_csv(StringData, sep =",")
        ##############################data analysis#####################################
        lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R = ms.logged_data_analysis_multimove(df,base2_R,base2_p,realrobot=False)
        
        this_timestamp=[]
        this_curve_dx=[]
        this_curve_dy=[]
        this_curve_dz=[]
        
        for ti in range(len(timestamp_prev)):
            t=timestamp_prev[ti]
            if t not in timestamp:
                continue
            curve_i = np.argwhere(timestamp==t)[0][0]
            this_timestamp.append(t)
            this_curve_dx.append(curve_exe1[curve_i][0]-curve_prev[ti][0])
            this_curve_dy.append(curve_exe1[curve_i][1]-curve_prev[ti][1])
            this_curve_dz.append(curve_exe1[curve_i][2]-curve_prev[ti][2])
        timestamp_xyz.append(np.array(this_timestamp))
        curve_xyz_dx.append(this_curve_dx)
        curve_xyz_dy.append(this_curve_dy)
        curve_xyz_dz.append(this_curve_dz)
        
    ## draw dx dy dz
    # dx
    marker_size=2
    all_title=['du_i [e 0 0]','du_i [0 e 0]','du_i [0 0 e]']
    for u_pos_i in range(3):
        fig, ax = plt.subplots(3,1)

        timestamp_prev[0]=timestamp_xyz[u_pos_i][0]
        timestamp_prev[-1]=timestamp_xyz[u_pos_i][-1]
        dx_interp = interp1d(timestamp_xyz[u_pos_i],curve_xyz_dx[u_pos_i],kind='cubic')(timestamp_prev)
        ax[0].plot(timestamp_xyz[u_pos_i],curve_xyz_dx[u_pos_i],'-bo',markersize=marker_size) # x deviation
        ax[0].plot(timestamp_prev,dx_interp,'-go',markersize=marker_size)
        ax[0].set_title('traj new, x deviation')

        dy_interp = interp1d(timestamp_xyz[u_pos_i],curve_xyz_dy[u_pos_i],kind='cubic')(timestamp_prev)
        ax[1].plot(timestamp_xyz[u_pos_i],curve_xyz_dy[u_pos_i],'-bo',markersize=marker_size) # y deviation
        ax[1].plot(timestamp_prev,dy_interp,'-go',markersize=marker_size)
        ax[1].set_title('traj new, y deviation')

        dz_interp = interp1d(timestamp_xyz[u_pos_i],curve_xyz_dz[u_pos_i],kind='cubic')(timestamp_prev)
        ax[2].plot(timestamp_xyz[u_pos_i],curve_xyz_dz[u_pos_i],'-bo',markersize=marker_size) # z deviation
        ax[2].plot(timestamp_prev,dz_interp,'-go',markersize=marker_size)
        ax[2].set_title('traj new, z deviation')

        if len(np.argwhere(timestamp_xyz[u_pos_i]==peak_time)) != 0:
            peak_i = np.argwhere(timestamp_xyz[u_pos_i]==peak_time)[0,0]
            ax[0].scatter(timestamp_xyz[u_pos_i][peak_i],curve_xyz_dx[u_pos_i][peak_i])
            ax[1].scatter(timestamp_xyz[u_pos_i][peak_i],curve_xyz_dy[u_pos_i][peak_i])
            ax[2].scatter(timestamp_xyz[u_pos_i][peak_i],curve_xyz_dz[u_pos_i][peak_i])

        fig.suptitle(all_title[u_pos_i]+' e='+str(epsilon))
        plt.show()

    # 
    # p_bp1_closest = deepcopy(p_bp1[closest_bp_id][-1])
    # p_bp1_second_closest = deepcopy(p_bp1[second_closest_bp_id][-1])
    # # calculate desired robot1 point
    # _,peak_error_curve_idx=calc_error(relative_path_exe[the_peak],relative_path[:,:3])  # index of original curve closest to max error point
    # tool_in_base1 = rox.Transform(base2_R,base2_p)*robot2.fwd(curve_exe_js2[the_peak])
    # desire_p1 = tool_in_base1*rox.Transform(np.eye(3),relative_path[peak_error_curve_idx,:3]) # desire p at time t
    # actual_p1 = robot1.fwd(curve_exe_js1[the_peak])
    # # x dir
    # error_dir1 = (desire_p1.p-actual_p1.p)/np.linalg.norm(desire_p1.p-actual_p1.p)
    # # y dir
    # if closest_bp_id>second_closest_bp_id:
    #     bp_dir1 = (p_bp1_closest-p_bp1_second_closest)/np.linalg.norm(p_bp1_closest-p_bp1_second_closest)
    # else:
    #     bp_dir1 = (p_bp1_second_closest-p_bp1_closest)/np.linalg.norm(p_bp1_second_closest-p_bp1_closest)
    # bp_dir1 = bp_dir1-np.dot(bp_dir1,error_dir1)*error_dir1
    # bp_dir1 = bp_dir1/np.linalg.norm(bp_dir1)
    # # z dir
    # outplane_dir1 = np.cross(error_dir1,bp_dir1)
    # outplane_dir1 = outplane_dir1/np.linalg.norm(outplane_dir1)



if __name__ == "__main__":
    main()