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
from copy import deepcopy
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
    # data_type='curve_2_scale'

    # data and curve directory
    curve_data_dir='../../../data/'+data_type+'/'

    test_type='50L'
    # test_type='30L'
    # test_type='greedy0.2'
    # test_type='greedy0.02'
    # test_type='moveLgreedy0.2'
    # test_type='moveLgreedy0.02'
    # test_type='minStepMoveLgreedy0.2'
    # test_type='minStepgreedy0.02'
    # test_type='minStepgreedy0.2'

    cmd_dir='../data/'+data_type+'/dual_arm_de/'+test_type+'/'
    # cmd_dir='../data/'+data_type+'/dual_arm_de_possibilyimpossible/'+test_type+'/'

    # relative path
    relative_path = read_csv(curve_data_dir+"/Curve_dense.csv", header=None).values

    H_200id=np.loadtxt(cmd_dir+'../base.csv',delimiter=',')
    base2_R=H_200id[:3,:3]
    base2_p=H_200id[:-1,-1]
    # print(base2_R)
    print(base2_p)
    k,theta=R2rot(base2_R)
    print(k,np.degrees(theta))

    robot2_tcp=np.loadtxt(cmd_dir+'../tcp.csv',delimiter=',')
    # print(robot2_tcp)

    # exit()
    
    ## robot
    toolbox_path = '../../../toolbox/'
    robot1 = robot_obj('FANUC_m10ia',toolbox_path+'robot_info/fanuc_m10ia_robot_default_config.yml',tool_file_path=toolbox_path+'tool_info/paintgun.csv',d=50,acc_dict_path=toolbox_path+'robot_info/m10ia_acc_compensate.pickle',j_compensation=[1,1,-1,-1,-1,-1])
    robot2=robot_obj('FANUC_lrmate200id',toolbox_path+'robot_info/fanuc_lrmate200id_robot_default_config.yml',tool_file_path=cmd_dir+'../tcp.csv',base_transformation_file=cmd_dir+'../base.csv',acc_dict_path=toolbox_path+'robot_info/lrmate200id_acc_compensate.pickle',j_compensation=[1,1,-1,-1,-1,-1])
    
    # robot2=lrmate200id(R_tool=Ry(np.pi/2)@robot2_tcp[:3,:3],p_tool=Ry(np.pi/2)@robot2_tcp[:3,-1])
    # print(robot2.fwd(np.radians([-41.52,-12.97,156.93,-57.35,-31.39,97.72])))
    # print(robot2.fwd(np.radians([0,0,0,0,0,0])))
    # exit()

    # fanuc motion send tool
    if data_type=='curve_1':
        ms = MotionSendFANUC(robot1=robot1,robot2=robot2)
    elif data_type=='curve_2_scale':
        ms = MotionSendFANUC(robot1=robot1,robot2=robot2,utool2=3)

    s=500 # mm/sec in leader frame
    z=100 # CNT100

    breakpoints1,primitives1,p_bp1,q_bp1,_=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command1.csv')
    breakpoints2,primitives2,p_bp2,q_bp2,_=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command2.csv')

    ###extension
    q_bp1_origin=deepcopy(q_bp1)
    q_bp2_origin=deepcopy(q_bp2)
    # print(np.degrees(q_bp2[0][-1]))
    # p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1,0,extension_start=25,extension_end=120)  ## curve_1, movel
    # p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1,0,extension_start=25,extension_end=50)  ## curve_1, movel
    p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1,0,extension_start=25,extension_end=50)  ## curve_1, greedy 0.2
    # p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1,0,extension_start=2,extension_end=120)  ## curve_1, movel greedy 0.02
    
    # p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1,0,extension_start=60,extension_end=150) ## curve_2_scale, movel
    # p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1,0,extension_start=60,extension_end=120) ## curve_2_scale, movel greedy 0.2
    # p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1,0,extension_start=50,extension_end=120) ## curve_2_scale, movel greedy 0.02
    # p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1,0,extension_start=50,extension_end=120) ## curve_2_scale, greedy 0.2
    # p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1,0,extension_start=30,extension_end=80) ## curve_2_scale, greedy 0.02
    # p_bp1,q_bp1,p_bp2,q_bp2=ms.extend_dual(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1,0,extension_start=25,extension_end=50) ## curve_2_scale, minstep greedy 0.02
    
    # q_bp1_origin_flat=[]
    # q_bp2_origin_flat=[]
    # q_bp1_flat=[]
    # q_bp2_flat=[]
    # for i in range(len(q_bp1_origin)):
    #     for j in range(len(q_bp1_origin[i])):
    #         q_bp1_origin_flat.append(q_bp1_origin[i][j])
    #         q_bp2_origin_flat.append(q_bp2_origin[i][j])
    #         q_bp1_flat.append(q_bp1[i][j])
    #         q_bp2_flat.append(q_bp2[i][j])

    # p_bp_relative,_=ms.form_relative_path(q_bp1_origin_flat,q_bp2_origin_flat,base2_R,base2_p)
    # p_bp_relative_new,_=ms.form_relative_path(q_bp1_flat,q_bp2_flat,base2_R,base2_p)
    # ### update visualization
    # ax = plt.axes(projection='3d')
    # ax.plot3D(p_bp_relative[:,0], p_bp_relative[:,1],p_bp_relative[:,2], 'red')
    # ax.scatter3D(p_bp_relative[:,0], p_bp_relative[:,1],p_bp_relative[:,2], 'red')
    # ax.plot3D(p_bp_relative_new[:,0], p_bp_relative_new[:,1],p_bp_relative_new[:,2], 'blue')
    # ax.scatter3D(p_bp_relative_new[:,0], p_bp_relative_new[:,1],p_bp_relative_new[:,2], 'blue')
    # plt.show()
    # exit()


    # print(robot2.fwd(q_bp2[0][0]))
    # print(robot2.inv(robot2.fwd(q_bp2[0][0]).p,robot2.fwd(q_bp2[0][0]).R,q_bp2[1][0]))
    # print(robot2.inv(np.array([-646.2,-504.9,1099]),robot2.fwd(q_bp2[0][0]).R,q_bp2[1][0]))
    # # print(q_bp2)
    # exit()
    print(len(q_bp1))
    print(len(q_bp2))
    print(len(primitives1))
    print(len(primitives2))

    # breakpoints1,primitives1,p_bp1,q_bp1,_=ms.extract_data_from_cmd(os.getcwd()+'/'+ilc_output+'command_arm1_9.csv')
    # breakpoints2,primitives2,p_bp2,q_bp2,_=ms.extract_data_from_cmd(os.getcwd()+'/'+ilc_output+'command_arm2_9.csv')
    # print(len(q_bp1))
    # print(len(q_bp2))
    # print(len(primitives1))
    # print(len(primitives2))

    # print(robot1.jacobian(q_bp1[0][0]))
    # for i in range(len(q_bp1)):
    #     u,sv,v=np.linalg.svd(robot1.jacobian(q_bp1[i][0]))
    #     print(np.min(sv))
    #     u,sv,v=np.linalg.svd(robot2.jacobian(q_bp2[i][0]))
    #     print(np.min(sv))
    #     print('======================')
    # exit()

    ###ilc toolbox def
    ilc=ilc_toolbox([robot1,robot2],[primitives1,primitives2])

    max_error_tolerance=0.5
    max_ang_error_tolerance=np.radians(3)

    multi_peak_threshold=0.2
    ###TODO: extension fix start point, moveC support
    iteration=30
    draw_speed_max=None
    draw_error_max=None
    max_error = 999
    error_localmin_flag=False
    use_grad=False
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

        ##### draw bp #####
        draw_moveC=False
        draw_moveL=False
        dlam=lam[-1]/breakpoints1[-1]
        for bpi in range(len(p_bp1)-3):
            bp_num=breakpoints1[bpi]
            bp_lam=bp_num*dlam
            if primitives1[bpi+1]=='movec_fit':
                if draw_moveC:
                    plt.axvline(x = bp_lam, color = 'c')
                else:
                    plt.axvline(x = bp_lam, color = 'c',label='moveC')
                    draw_moveC=True
            elif primitives1[bpi+1]=='movel_fit':
                if draw_moveL:
                    plt.axvline(x = bp_lam, color = 'm')
                else:
                    plt.axvline(x = bp_lam, color = 'm',label='moveL')
                    draw_moveL=True
        bp_num=breakpoints1[-1]
        bp_lam=bp_num*dlam
        ###################

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

        ilc_output=cmd_dir+'results_'+str(s)+'_'+test_type+'/'
        Path(ilc_output).mkdir(exist_ok=True)
        plt.savefig(ilc_output+'iteration_ '+str(i))
        plt.clf()
        # # plt.show()
        # exit()

        df=DataFrame({'primitives':primitives1,'points':p_bp1,'q_bp':q_bp1})
        df.to_csv(ilc_output+'command_arm1_'+str(i)+'.csv',header=True,index=False)
        df=DataFrame({'primitives':primitives2,'points':p_bp2,'q_bp':q_bp2})
        df.to_csv(ilc_output+'command_arm2_'+str(i)+'.csv',header=True,index=False)

        ###########################plot for verification###################################
        # p_bp_relative,_=ms.form_relative_path(np.squeeze(q_bp1),np.squeeze(q_bp2),base2_R,base2_p)

        if max(error)>max_error and error_localmin_flag:
            print("Can't decrease anymore")
            break

        if max(error)>max_error:
            print("Use grad")
            error_localmin_flag=True
            use_grad=True
        
        if max(error)<max_error:
            error_localmin_flag=False

        max_error=max(error)

        if max(error) < max_error_tolerance and max(angle_error)<max_ang_error_tolerance:
            time.sleep(1)
            break

        if use_grad:
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
                # q_bp1_origin_flat=[]
                # q_bp2_origin_flat=[]
                # q_bp1_flat=[]
                # q_bp2_flat=[]
                # for i in range(len(q_bp1_origin)):
                #     for j in range(len(q_bp1_origin[i])):
                #         q_bp1_origin_flat.append(q_bp1_origin[i][j])
                #         q_bp2_origin_flat.append(q_bp2_origin[i][j])
                #         q_bp1_flat.append(q_bp1[i][j])
                #         q_bp2_flat.append(q_bp2[i][j])

                # p_bp_relative_new,_=ms.form_relative_path(np.squeeze(q_bp1_new),np.squeeze(q_bp2_new),base2_R,base2_p)

                # for bp_new in p_bp_relative_new[breakpoint_interp_2tweak_indices]:
                #     all_new_bp.append(bp_new)
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
        else:
            error_bps_v1,error_bps_w1,error_bps_v2,error_bps_w2=ilc.get_error_direction_dual(relative_path,p_bp1,q_bp1,p_bp2,q_bp2,relative_path_exe,relative_path_exe_R,curve_exe1,curve_exe_R1,curve_exe2,curve_exe_R2)
            error_bps_w1=np.zeros(error_bps_w1.shape)
            error_bps_w2=np.zeros(error_bps_w2.shape)
            # error_bps_v1=np.zeros(error_bps_v1.shape)
            # error_bps_v2=np.zeros(error_bps_v2.shape)

            gamma_v=0.2
            gamma_w=0.1
            p_bp1_origin=deepcopy(p_bp1)
            p_bp2_origin=deepcopy(p_bp2)
            q_bp1_origin=deepcopy(q_bp1)
            q_bp2_origin=deepcopy(q_bp2)
            while True:
                try:
                    p_bp1, q_bp1, p_bp2, q_bp2=ilc.update_error_direction_dual(relative_path,p_bp1_origin,q_bp1_origin,p_bp2_origin,q_bp2_origin,error_bps_v1,error_bps_w1,error_bps_v2,error_bps_w2,gamma_v,gamma_w)
                    break
                except IndexError:
                    gamma_v*=0.75
                    gamma_w*=0.75
                    if gamma_w<0.02:
                        # error_localmin_flag=True
                        use_grad=True
                        break

if __name__ == "__main__":
    main()