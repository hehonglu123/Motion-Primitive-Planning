########
# This module utilized https://github.com/johnwason/abb_motion_program_exec
# and send whatever the motion primitives that algorithms generate
# to RobotStudio
########

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

def main():
    # curve
    data_type='blade'
    # data_type='wood'

    # data and curve directory
    if data_type=='blade':
        curve_data_dir='../../../data/from_NX/'
        cmd_dir='../data/curve_blade/'
        data_dir='data/blade_dual/'
    elif data_type=='wood':
        curve_data_dir='../../../data/wood/'
        cmd_dir='../data/curve_wood/'
        data_dir='data/wood_dual/'

    test_type='dual_arm'
    # test_type='dual_single_arm'
    # test_type='dual_single_arm_straight' # robot2 is multiple user defined straight line
    # test_type='dual_single_arm_straight_50' # robot2 is multiple user defined straight line
    # test_type='dual_single_arm_straight_min' # robot2 is multiple user defined straight line
    # test_type='dual_single_arm_straight_min10' # robot2 is multiple user defined straight line
    # test_type='dual_arm_10'

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

    # fanuc motion send tool
    if data_type=='blade':
        ms = MotionSendFANUC(robot1=robot1,robot2=robot2)
    elif data_type=='wood':
        ms = MotionSendFANUC(robot1=robot1,robot2=robot2,utool2=3)

    # s=int(1600/2.) # mm/sec in leader frame
    s=12 # mm/sec in leader frame
    z=100 # CNT100
    ilc_output=data_dir+'results_'+str(s)+'_'+test_type+'/'
    Path(ilc_output).mkdir(exist_ok=True)

    breakpoints1,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command1.csv')
    breakpoints2,primitives2,p_bp2,q_bp2=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command2.csv')

    q_bp1_start = q_bp1[0][0]
    q_bp1_end = q_bp1[-1][-1]
    q_bp2_start = q_bp2[0][0]
    q_bp2_end = q_bp2[-1][-1]

    p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1,base2_T,extension_d=300)

    ## calculate step at start and end
    step_start=None
    step_end=None
    for i in range(len(q_bp1)):
        if np.all(q_bp1[i][0]==q_bp1_start):
            step_start=i
        if np.all(q_bp1[i][-1]==q_bp1_end):
            step_end=i

    assert step_start is not None,'Cant find step start'
    assert step_end is not None,'Cant find step start'
    print(step_start,step_end)

    step_start=None
    step_end=None
    for i in range(len(q_bp2)):
        if np.all(q_bp2[i][0]==q_bp2_start):
            step_start=i
        if np.all(q_bp2[i][-1]==q_bp2_end):
            step_end=i

    assert step_start is not None,'Cant find step start'
    assert step_end is not None,'Cant find step start'
    print(step_start,step_end)

    use_exist=False
    use_iteration=231
    if use_exist:
        _,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(os.getcwd()+'/'+ilc_output+'command_arm1_'+str(use_iteration)+'.csv')
        _,primitives2,p_bp2,q_bp2=ms.extract_data_from_cmd(os.getcwd()+'/'+ilc_output+'command_arm2_'+str(use_iteration)+'.csv')

    # for i in range(len(primitives1)):
    #     primitives1[i]='movej_fit'
    #     primitives2[i]='movej_fit'

    ###ilc toolbox def
    ilc=ilc_toolbox([robot1,robot2],[primitives1,primitives2],base2_R,base2_p)

    multi_peak_threshold=0.2
    ###TODO: extension fix start point, moveC support
    iteration=500
    draw_speed_max=None
    draw_error_max=None
    max_error_tolerance = 0.5
    start_iteration=0
    if use_exist:
        start_iteration=use_iteration
    for i in range(start_iteration,iteration):

        ###execution with plant
        logged_data=ms.exec_motions_multimove_nocoord(robot1,robot2,primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,s,s,z,z)
        # with open('iteration_'+str(i)+'.csv',"wb") as f:
        #     f.write(logged_data)
        StringData=StringIO(logged_data.decode('utf-8'))
        df = read_csv(StringData, sep =",")
        ##############################data analysis#####################################
        lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R = ms.logged_data_analysis_multimove(df,base2_R,base2_p,realrobot=False)
        #############################chop extension off##################################
        lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
            ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])
        ave_speed=np.mean(speed)

        ##############################calcualte error########################################
        error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])
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
        plt.savefig(ilc_output+'iteration_ '+str(i))
        plt.clf()
        # plt.show()
        # exit()

        df=DataFrame({'primitives':primitives1,'points':p_bp1,'q_bp':q_bp1})
        df.to_csv(ilc_output+'command_arm1_'+str(i)+'.csv',header=True,index=False)
        df=DataFrame({'primitives':primitives2,'points':p_bp2,'q_bp':q_bp2})
        df.to_csv(ilc_output+'command_arm2_'+str(i)+'.csv',header=True,index=False)

        ##########################################calculate error direction and push######################################
        curve_target = np.zeros((len(relative_path_exe), 3))
        curve_target_R = np.zeros((len(relative_path_exe), 3))
        error_target_dir = np.zeros(len(relative_path_exe),3)
        for i in range(len(relative_path_exe)):
            dist = np.linalg.norm(relative_path[:,:3] - relative_path_exe[i], axis=1)
            closest_point_idx = np.argmin(dist)
            curve_target[i, :] = relative_path[closest_point_idx, :3]
            curve_target_R[i, :] = relative_path[closest_point_idx, 3:]
            error_target_dir[i,:] = relative_path_exe[i] - curve_target[i, :]
        
        p_bp_relative,_=ms.form_relative_path(np.squeeze(q_bp1),np.squeeze(q_bp2),base2_R,base2_p)
        p_bp1_error_dir=[]
        # find error direction
        for i in range(1, len(p_bp_relative) - 1):
            p_bp = p_bp_relative[i][-1]
            closest_point_idx = np.argmin(np.linalg.norm(curve_target - p_bp, axis=1))
            error_dir = error[error_target_dir]
            p_bp1_error_dir.append(error_dir)
        p_bp1_error_dir.append(np.array([0,0,0])) # no update on the end breakpoints
        p_bp1_error_dir.insert(0,np.array([0,0,0])) # no update on the first breakpoints
        
        ### update p_bp1
        for i in range(0,len(p_bp1)):
            p_bp1[i][-1] = np.array(p_bp1[i][-1]) - p_bp1_error_dir[i]

        p_bp_relative_new,_=ms.form_relative_path(np.squeeze(q_bp1),np.squeeze(q_bp2),base2_R,base2_p)
        ### update visualization
        ax = plt.axes(projection='3d')
        ax.plot3D(relative_path[:,0], relative_path[:,1],relative_path[:,2], 'red',label='original')
        # ax.scatter3D(p_bp_relative[step_start:step_end+1,0], p_bp_relative[step_start:step_end+1,1],p_bp_relative[step_start:step_end+1,2], 'blue', label='breakpoints')
        ax.scatter3D(relative_path_exe[peaks,0], relative_path_exe[peaks,1], relative_path_exe[peaks,2],c='orange',label='worst case')
        # ax.scatter3D(all_new_bp[:,0], all_new_bp[:,1], all_new_bp[:,2], c='magenta',label='new breakpoints')
        ax.plot3D(relative_path_exe[:,0], relative_path_exe[:,1],relative_path_exe[:,2], 'green',label='execution')
        ax.view_init(61, -67)
        plt.legend()
        # plt.savefig(ilc_output+'traj_iteration_'+str(i))
        # plt.clf()
        # plt.show()



if __name__ == "__main__":
    main()