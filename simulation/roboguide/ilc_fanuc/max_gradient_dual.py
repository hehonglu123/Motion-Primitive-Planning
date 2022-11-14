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
    data_type='curve_1'
    data_type='curve_2_scale'

    # data and curve directory
    curve_data_dir='../../../data/'+data_type+'/'

    test_type='30'

    cmd_dir='../data/'+data_type+'/dual_arm_de/'+test_type+'/'

    # relative path
    relative_path = read_csv(curve_data_dir+"/Curve_dense.csv", header=None).values

    H_200id=np.loadtxt(cmd_dir+'../base.csv',delimiter=',')
    base2_R=H_200id[:3,:3]
    base2_p=H_200id[:-1,-1]

    print(base2_R)
    print(base2_p)
    k,theta=R2rot(base2_R)
    print(k,np.degrees(theta))

    exit()

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

    s=1500 # mm/sec in leader frame
    z=100 # CNT100
    ilc_output=data_dir+'results_'+str(s)+'_'+test_type+'/'
    Path(ilc_output).mkdir(exist_ok=True)

    breakpoints1,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command1.csv')
    breakpoints2,primitives2,p_bp2,q_bp2=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command2.csv')

    q_bp1_start = q_bp1[0][0]
    q_bp1_end = q_bp1[-1][-1]

    ###extension
    p_bp1,q_bp1,p_bp2,q_bp2,step_to_extend_end=ms.extend_dual_relative(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1,base2_T,extension_d=300)

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

    ###ilc toolbox def
    ilc=ilc_toolbox([robot1,robot2],[primitives1,primitives2],base2_R,base2_p)

    multi_peak_threshold=0.2
    ###TODO: extension fix start point, moveC support
    iteration=500
    draw_speed_max=None
    draw_error_max=None
    max_error_tolerance = 0.5
    for i in range(iteration):

        ###execution with plant
        logged_data=ms.exec_motions_multimove(robot1,robot2,primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,s,z)
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

        # plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot3D(relative_path[:,0], relative_path[:,1],relative_path[:,2], 'red',label='original')
        # ax.scatter3D(relative_path[breakpoints1[:-1],0], relative_path[breakpoints1[:-1],1],relative_path[breakpoints1[:-1],2], 'blue')
        # ax.scatter3D(relative_path_exe[peaks,0], relative_path_exe[peaks,1], relative_path_exe[peaks,2],c='orange',label='worst case')
        # ax.plot3D(relative_path_exe[:,0], relative_path_exe[:,1],relative_path_exe[:,2], 'green',label='execution')
        # ax.view_init(100, -101)
        # plt.legend()
        # plt.savefig(ilc_output+'traj_iteration_ '+str(i))
        # plt.clf()
        # plt.show()

        # if i%10==0:
        #     df=DataFrame({'primitives':primitives1,'points':p_bp1,'q_bp':q_bp1})
        #     df.to_csv(ilc_output+'command_arm1_'+str(i)+'.csv',header=True,index=False)
        #     df=DataFrame({'primitives':primitives2,'points':p_bp2,'q_bp':q_bp2})
        #     df.to_csv(ilc_output+'command_arm2_'+str(i)+'.csv',header=True,index=False)
        # if max(error) < max_error_tolerance:
        #     df=DataFrame({'primitives':primitives1,'points':p_bp1,'q_bp':q_bp1})
        #     df.to_csv(ilc_output+'command_arm1_'+str(i)+'.csv',header=True,index=False)
        #     df=DataFrame({'primitives':primitives2,'points':p_bp2,'q_bp':q_bp2})
        #     df.to_csv(ilc_output+'command_arm2_'+str(i)+'.csv',header=True,index=False)
        #     break

        df=DataFrame({'primitives':primitives1,'points':p_bp1,'q_bp':q_bp1})
        df.to_csv(ilc_output+'command_arm1_'+str(i)+'.csv',header=True,index=False)
        df=DataFrame({'primitives':primitives2,'points':p_bp2,'q_bp':q_bp2})
        df.to_csv(ilc_output+'command_arm2_'+str(i)+'.csv',header=True,index=False)

        ###########################plot for verification###################################
        
        p_bp_relative,_=ms.form_relative_path(np.squeeze(q_bp1),np.squeeze(q_bp2),base2_R,base2_p)
        # ax = plt.axes(projection='3d')
        # ax.plot3D(relative_path[:,0], relative_path[:,1],relative_path[:,2], 'red',label='original')
        # ax.scatter3D(p_bp_relative[step_start:step_end+1,0], p_bp_relative[step_start:step_end+1,1],p_bp_relative[step_start:step_end+1,2], 'blue', label='breakpoints')
        # ax.scatter3D(relative_path_exe[peaks,0], relative_path_exe[peaks,1], relative_path_exe[peaks,2],c='orange',label='worst case')
        # ax.scatter3D(p_bp_relative_new[breakpoint_interp_2tweak_indices,0], p_bp_relative_new[breakpoint_interp_2tweak_indices,1], p_bp_relative_new[breakpoint_interp_2tweak_indices,2], c='magenta',label='new breakpoints')
        # ax.plot3D(relative_path_exe[:,0], relative_path_exe[:,1],relative_path_exe[:,2], 'green',label='execution')
        # ax.view_init(100, -101)
        # plt.legend()
        # plt.savefig(ilc_output+'traj_iteration_ '+str(i))
        # plt.clf()
        # plt.show()

        ##########################################calculate gradient for peaks######################################
        ###restore trajectory from primitives
        curve_interp1, curve_R_interp1, curve_js_interp1, breakpoints_blended=form_traj_from_bp(q_bp1,primitives1,robot1)
        curve_interp2, curve_R_interp2, curve_js_interp2, breakpoints_blended=form_traj_from_bp(q_bp2,primitives2,robot2)
        curve_js_blended1,curve_blended1,curve_R_blended1=blend_js_from_primitive(curve_interp1, curve_js_interp1, breakpoints_blended, primitives1,robot1,zone=10)
        curve_js_blended2,curve_blended2,curve_R_blended2=blend_js_from_primitive(curve_interp2, curve_js_interp2, breakpoints_blended, primitives2,robot2,zone=10)

        ###establish relative trajectory from blended trajectory
        relative_path_blended,relative_path_blended_R=ms.form_relative_path(curve_js_blended1,curve_js_blended2,base2_R,base2_p)

        all_new_bp=[]
        for peak in peaks:
            ######gradient calculation related to nearest 3 points from primitive blended trajectory, not actual one
            _,peak_error_curve_idx=calc_error(relative_path_exe[peak],relative_path[:,:3])  # index of original curve closest to max error point

            ###get closest to worst case point on blended trajectory
            _,peak_error_curve_blended_idx=calc_error(relative_path_exe[peak],relative_path_blended)

            ###############get numerical gradient#####
            ###find closest 3 breakpoints
            order=np.argsort(np.abs(breakpoints_blended-peak_error_curve_blended_idx))
            breakpoint_interp_2tweak_indices=order[:3]

            de_dp=ilc.get_gradient_from_model_xyz_dual(\
                [p_bp1,p_bp2],[q_bp1,q_bp2],breakpoints_blended,[curve_blended1,curve_blended2],peak_error_curve_blended_idx,[curve_exe_js1[peak],curve_exe_js2[peak]],relative_path[peak_error_curve_idx,:3],breakpoint_interp_2tweak_indices)


            p_bp1_new, q_bp1_new,p_bp2_new,q_bp2_new=ilc.update_bp_xyz_dual([p_bp1,p_bp2],[q_bp1,q_bp2],de_dp,error[peak],breakpoint_interp_2tweak_indices,alpha=0.5)

            #########plot adjusted breakpoints
            p_bp_relative_new,_=ms.form_relative_path(np.squeeze(q_bp1_new),np.squeeze(q_bp2_new),base2_R,base2_p)

            for bp_new in p_bp_relative_new[breakpoint_interp_2tweak_indices]:
                all_new_bp.append(bp_new)
            # print(all_new_bp)


            # ax.scatter3D(p_bp_relative_new[breakpoint_interp_2tweak_indices,0], p_bp_relative_new[breakpoint_interp_2tweak_indices,1], p_bp_relative_new[breakpoint_interp_2tweak_indices,2], c='blue',label='new breakpoints')
            # plt.legend()
            # plt.show()
            
            # plt.show()

            ###update
            p_bp1=p_bp1_new
            q_bp1=q_bp1_new
            p_bp2=p_bp2_new
            q_bp2=q_bp2_new
        
        all_new_bp=np.array(all_new_bp)
        ax = plt.axes(projection='3d')
        ax.plot3D(relative_path[:,0], relative_path[:,1],relative_path[:,2], 'red',label='original')
        ax.scatter3D(p_bp_relative[step_start:step_end+1,0], p_bp_relative[step_start:step_end+1,1],p_bp_relative[step_start:step_end+1,2], 'blue', label='breakpoints')
        ax.scatter3D(relative_path_exe[peaks,0], relative_path_exe[peaks,1], relative_path_exe[peaks,2],c='orange',label='worst case')
        ax.scatter3D(all_new_bp[:,0], all_new_bp[:,1], all_new_bp[:,2], c='magenta',label='new breakpoints')
        ax.plot3D(relative_path_exe[:,0], relative_path_exe[:,1],relative_path_exe[:,2], 'green',label='execution')
        ax.view_init(100, -101)
        plt.legend()
        plt.savefig(ilc_output+'traj_iteration_ '+str(i))
        plt.clf()



if __name__ == "__main__":
    main()