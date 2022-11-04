import numpy as np
from math import degrees,radians,ceil,floor
import yaml
from copy import deepcopy
from scipy.signal import find_peaks
from general_robotics_toolbox import *
from pandas import read_csv,DataFrame
import sys
from io import StringIO
from matplotlib import pyplot as plt
from pathlib import Path
import os
from fanuc_motion_program_exec_client import *

# sys.path.append('../abb_motion_program_exec')
# from abb_motion_program_exec_client import *
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

    # all_objtype=['wood','blade_scale']
    # all_objtype=['blade_scale']
    # all_objtype=['wood']
    all_objtype=['curve_blade_scale']
    # all_objtype=['curve_wood']

    # num_ls=[80,100,150]
    # num_ls=[100,50,30,25]
    num_ls=[25]

    for obj_type in all_objtype:

        # obj_type='wood'
        # obj_type='blade'
        print(obj_type)
        
        # data_dir='../data/baseline_m10ia/'+obj_type+'/'
        data_dir='../data/'+obj_type+'/single_arm_de/'

        robot=m10ia(d=50)
        ms = MotionSendFANUC()
        multi_peak_threshold=0.2
        curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
        curve = np.array(curve)
        curve_normal = curve[:,3:]
        # curve = curve[:,:3]

        for num_l in num_ls:
            print(obj_type+' '+str(num_l))

            # cmd_dir='../data/baseline_m10ia/'+obj_type+'/'+str(num_l)+'/'
            cmd_dir=data_dir+str(num_l)+'/'

            try:
                breakpoints,primitives,p_bp,q_bp,_=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command.csv')
                # breakpoints,primitives,p_bp,q_bp=extract_data_from_cmd(os.getcwd()+'/'+ilc_output+'command_25.csv')
            except:
                print("Convert bp to command")
                exit()

            q_bp_start = q_bp[0][0]
            q_bp_end = q_bp[-1][-1]
            primitives,p_bp,q_bp=ms.extend_start_end(robot,q_bp,primitives,breakpoints,p_bp,extension_start=100,extension_end=100)
            # primitives,p_bp,q_bp=ms.extend_start_end_qp(robot,q_bp,primitives,breakpoints,p_bp,extension_start=300,extension_end=300)

            p_bp_origin = deepcopy(p_bp)
            q_bp_origin = deepcopy(q_bp)

            ## calculate step at start and end
            step_start1=None
            step_end1=None
            for i in range(len(q_bp)):
                if np.all(q_bp[i][0]==q_bp_start):
                    step_start1=i
                if np.all(q_bp[i][-1]==q_bp_end):
                    step_end1=i

            assert step_start1 is not None,'Cant find step start'
            assert step_end1 is not None,'Cant find step start'
            print(step_start1,step_end1)

            ###### bi-section searching speed to satisfied speed variance
            s_up=1000
            s_low=0
            s=(s_up+s_low)/2
            speed_found=False
            z=100
            while True:
                s=int(s)
                
                logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,s,z)
                StringData=StringIO(logged_data.decode('utf-8'))
                df = read_csv(StringData)
                lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df)
                lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[:,:3],curve_normal)
                speed_ave=np.mean(speed)
                speed_std=np.std(speed)
                error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve_normal)
                error_max=max(error)
                angle_error_max=max(angle_error)
                break
                if speed_found:
                    break

                print('Speed:',s,',Error:',error_max,',Angle Error:',degrees(angle_error_max),'Speed Ave:',speed_ave,'Speed Std:',speed_std)
                if speed_std>speed_ave*0.05:
                    s_up=s
                    s=ceil((s_up+s_low)/2.)
                else:
                    s_low=s
                    s=floor((s_up+s_low)/2.)
                    find_sol=True
                if s==s_low:
                    break
                elif s==s_up:
                    s=s_low
                    speed_found=True

            s_up=s
            s_low=30
            speed_found=False
            last_error = 999
            find_sol=False
            while True:
                alpha = 0.5 # for gradient descent
                alpha_error_dir = 0.8 # for pushing in error direction
                ilc_output=cmd_dir+'results_'+str(s)+'/'
                Path(ilc_output).mkdir(exist_ok=True)

                p_bp = deepcopy(p_bp_origin)
                q_bp = deepcopy(q_bp_origin)

                ###ilc toolbox def
                ilc=ilc_toolbox(robot,primitives)

                ###TODO: extension fix start point, moveC support
                iteration=100
                draw_speed_max=None
                draw_error_max=None
                max_error_tolerance = 0.5
                # max_error_all_thres = 1
                max_gradient_descent_flag = False
                max_error_prev = 999999999
                for i in range(iteration):
                    
                    ###execute,curve_fit_js only used for orientation
                    logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,s,z)

                    # with open(data_dir+"curve_js_iter0_exe.csv","wb") as f:
                    #     f.write(logged_data)
                    # exit()

                    # print(logged_data)
                    StringData=StringIO(logged_data.decode('utf-8'))
                    # df = read_csv(StringData, sep =",")
                    df = read_csv(StringData)
                    ##############################data analysis#####################################
                    lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df)

                    #############################chop extension off##################################
                    lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[:,:3],curve_normal)
                    ave_speed=np.mean(speed)

                    ##############################calcualte error########################################
                    error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve_normal)
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
                    
                    ##############################plot error#####################################
                    try:
                        plt.close(fig)
                    except:
                        pass
                    fig, ax1 = plt.subplots(figsize=(6,4))
                    ax2 = ax1.twinx()
                    ax1.plot(lam, speed, 'g-', label='Speed')
                    ax2.plot(lam, error, 'b-',label='Error')
                    ax2.scatter(lam[peaks],error[peaks],label='peaks')
                    ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
                    if draw_speed_max is None:
                        draw_speed_max=max(speed)*1.05
                    if draw_error_max is None:
                        draw_error_max=max(error)*1.05
                    ax1.axis(ymin=0,ymax=draw_speed_max)
                    ax2.axis(ymin=0,ymax=draw_error_max)

                    ax1.set_xlabel('lambda (mm)')
                    ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
                    ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
                    plt.title("Speed and Error Plot")
                    ax1.legend(loc=0)
                    ax2.legend(loc=0)

                    # save fig
                    plt.savefig(ilc_output+'iteration_'+str(i))
                    plt.clf()
                    # fig.canvas.manager.window.move(1300,99)
                    # plt.show(block=False)
                    # plt.pause(0.1)
                    # exit()
                    # save bp
                    df=DataFrame({'primitives':primitives,'points':p_bp,'q_bp':q_bp})
                    df.to_csv(ilc_output+'command_arm1_'+str(i)+'.csv',header=True,index=False)

                    if max(error) < max_error_tolerance:
                        time.sleep(5)
                        break
                    
                    # if max error does not decrease, and multi-peak max gradient descent not work
                    if max(error) > max_error_prev and max_gradient_descent_flag == True:
                        print("Can't decrease error anymore")
                        time.sleep(2)
                        break
                    # if max error does not decrease, use multi-peak max gradient descent
                    if max(error) > max_error_prev:
                        max_gradient_descent_flag = True

                    p_bp1_update = deepcopy(p_bp)
                    q_bp1_update = deepcopy(q_bp)
                    if not max_gradient_descent_flag: # update through push in error direction
                        ##########################################calculate error direction and push######################################
                        ### interpolate curve (get gradient direction)
                        curve_target = np.zeros((len(curve_exe), 3))
                        curve_target_R = np.zeros((len(curve_exe), 3))
                        for j in range(len(curve_exe)):
                            dist = np.linalg.norm(curve[:,:3] - curve_exe[j], axis=1)
                            closest_point_idx = np.argmin(dist)
                            curve_target[j, :] = curve[closest_point_idx, :3]
                            curve_target_R[j, :] = curve_normal[closest_point_idx]

                        ### get error (and transfer into robot1 frame)
                        error1 = []
                        angle_error1 = []
                        for j in range(len(curve_exe)):
                            ## calculate error
                            error1.append(curve_target[j]-curve_exe[j])
                            # angle_error1.append(get_angle(curve_exe_R1[j][:,-1], this_curve_target_R1))
                            angle_error1.append(curve_target_R[j]-curve_exe_R[j][:,-1])

                        ### get closets bp index
                        p_bp1_error_dir=[]
                        p_bp1_ang_error_dir=[]
                        # find error direction
                        for j in range(step_start1, step_end1+1): # exclude first and the last bp and the bp for extension
                            this_p_bp = p_bp[j]
                            closest_point_idx = np.argmin(np.linalg.norm(curve_target - this_p_bp, axis=1))
                            error_dir = error1[closest_point_idx]
                            p_bp1_error_dir.append(error_dir)
                            ang_error_dir = angle_error1[closest_point_idx]
                            p_bp1_ang_error_dir.append(ang_error_dir)
                        
                        # error_dir_toolbox=ilc.get_error_direction(curve,p_bp,q_bp,curve_exe,curve_exe_R)
                        # print(len(error_dir_toolbox))
                        # print(len(p_bp1_error_dir))
                        # print(len(p_bp1_ang_error_dir))
                        # for j in range(step_start1, step_end1+1):
                        #     print("BP:",j)
                        #     print("error dir:",np.dot(error_dir_toolbox[0][j][0]/np.linalg.norm(error_dir_toolbox[0][j][0]),p_bp1_error_dir[j-step_start1]/np.linalg.norm(p_bp1_error_dir[j-step_start1])))
                        #     print("ang error dir:",np.dot(error_dir_toolbox[1][j][0]/np.linalg.norm(error_dir_toolbox[1][j][0]),p_bp1_ang_error_dir[j-step_start1]/np.linalg.norm(p_bp1_ang_error_dir[j-step_start1])))
                        # exit()
                        
                        ### update all p_bp1
                        for j in range(step_start1,step_end1+1):
                            p_bp1_update[j][-1] = np.array(p_bp[j][-1]) + alpha_error_dir*p_bp1_error_dir[j-step_start1]
                            bp1_R = robot.fwd(q_bp[j][-1]).R
                            bp1_R[:,-1] = (bp1_R[:,-1] + alpha_error_dir*p_bp1_ang_error_dir[j-step_start1])/np.linalg.norm((bp1_R[:,-1] + alpha_error_dir*p_bp1_ang_error_dir[j-step_start1]))
                            q_bp1_update[j][-1] = car2js(robot, q_bp[j][-1], p_bp1_update[j][-1], bp1_R)[0]
                        
                        p_bp = deepcopy(p_bp1_update)
                        q_bp = deepcopy(q_bp1_update)
                    else: # update through multi-peak max gradient desecent
                        ##########################################calculate gradient######################################
                        ######gradient calculation related to nearest 3 points from primitive blended trajectory, not actual one
                        ###restore trajectory from primitives
                        curve_interp, curve_R_interp, curve_js_interp, breakpoints_blended=form_traj_from_bp(q_bp,primitives,robot)
                        curve_js_blended,curve_blended,curve_R_blended=blend_js_from_primitive(curve_interp, curve_js_interp, breakpoints_blended, primitives,robot,zone=10)
                        # fanuc blend in cart space
                        # curve_js_blended,curve_blended,curve_R_blended=blend_cart_from_primitive(curve_interp,curve_R_interp,curve_js_interp,breakpoints_blended,primitives,robot,ave_speed)
                        # plt.plot(curve_interp[:,0],curve_interp[:,1])
                        # plt.plot(curve_blended[:,0],curve_blended[:,1])
                        # plt.axis('equal')
                        # plt.show()
                        for peak in peaks:
                            ######gradient calculation related to nearest 3 points from primitive blended trajectory, not actual one
                            _,peak_error_curve_idx=calc_error(curve_exe[peak],curve[:,:3])  # index of original curve closest to max error point

                            ###get closest to worst case point on blended trajectory
                            _,peak_error_curve_blended_idx=calc_error(curve_exe[peak],curve_blended)

                            ###############get numerical gradient#####
                            ###find closest 3 breakpoints
                            order=np.argsort(np.abs(breakpoints_blended-peak_error_curve_blended_idx))
                            breakpoint_interp_2tweak_indices=order[:3]

                            de_dp=ilc.get_gradient_from_model_xyz_fanuc(p_bp,q_bp,breakpoints_blended,curve_blended,peak_error_curve_blended_idx,robot.fwd(curve_exe_js[peak]),curve[peak_error_curve_idx,:3],breakpoint_interp_2tweak_indices,ave_speed)
                            p_bp, q_bp=ilc.update_bp_xyz(p_bp,q_bp,de_dp,error[peak],breakpoint_interp_2tweak_indices,alpha=alpha)
                    
                    # update max error
                    max_error_prev = max(error)
                
                error_max = max(error)
                angle_error_max=max(angle_error)
                speed_ave = np.mean(speed)
                speed_std = np.std(speed)
                print('Speed:',s,',Error:',error_max,',Angle Error:',degrees(angle_error_max),'Speed Ave:',speed_ave,'Speed Std:',speed_std)
                if error_max > 0.5 or angle_error_max > radians(3) or speed_std>speed_ave*0.05:
                    s_up=s
                    s=ceil((s_up+s_low)/2.)
                else:
                    s_low=s
                    s=floor((s_up+s_low)/2.)
                    find_sol=True
                if s==s_low:
                    break
                elif s==s_up:
                    s=s_low
                    speed_found=True
                
                if not find_sol:
                    print("last error:",last_error,"this error:",error_max)
                    if last_error-error_max < 0.001:
                        print("Can't reduce error")
                        break
                last_error=error_max

if __name__ == "__main__":
    main()