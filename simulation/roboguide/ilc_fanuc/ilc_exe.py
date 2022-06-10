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
from matplotlib import pyplot as plt

# sys.path.append('../abb_motion_program_exec')
# from abb_motion_program_exec_client import *
sys.path.append('../fanuc_toolbox')
from fanuc_client import *
from fanuc_utils import *
sys.path.append('../../../ILC')
from ilc_toolbox import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
# from MotionSend import *
from lambda_calc import *
from blending import *

def main():
    # data_dir="fitting_output_new/python_qp_movel/"
    dataset='from_NX/'
    data_dir="../data/ilc_fanuc/"+dataset
    fitting_output='../data/ilc_fanuc/'+dataset
    ilc_output=fitting_output+'results/'


    curve_js=read_csv(data_dir+'Curve_js.csv',header=None).values
    curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values


    max_error_threshold=0.49
    robot=m710ic(d=50)

    s = 800
    z = 100

    # curve_fit_js=read_csv(fitting_output+'curve_fit_js.csv',header=None).values
    # ms = MotionSend()
    # breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd(fitting_output+'command.csv')
    ###extension
    # primitives,p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp)

    try:
        breakpoints,primitives,p_bp,q_bp=extract_data_from_cmd(fitting_output+'command.csv')
    except:
        print("exe")
        step=500
        total_seg = (len(curve_js)-1)/step
        breakpoints = [0]
        primitives = ['movej_fit']
        p_bp = [curve[0][:3]]
        q_bp = [[curve_js[0]]]
        for i in range(step,len(curve_js),step):
            breakpoints.append(i)
            primitives.append('movel_fit')
            p_bp.append(curve[i][:3])
            q_bp.append([curve_js[i]])
        
        df=DataFrame({'breakpoints':breakpoints,'primitives':primitives,'points':p_bp,'q_bp':q_bp})
        df.to_csv(fitting_output+'command.csv',header=True,index=False)

    primitives,p_bp,q_bp=extend_start_end(robot,q_bp,primitives,breakpoints,p_bp,extension_d=61)

    ###ilc toolbox def
    ilc=ilc_toolbox(robot,primitives)

    ###TODO: extension fix start point, moveC support
    max_error=999
    inserted_points=[]
    all_max_error=[]
    i=0
    iteration=0
    ms = MotionSendFANUC()
    while max_error>max_error_threshold:
        i+=1
        
        ###execute,curve_fit_js only used for orientation
        logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,s,z)

        # print(logged_data)
        StringData=StringIO(logged_data.decode('utf-8'))
        # df = read_csv(StringData, sep =",")
        df = read_csv(StringData)
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
        curve_exe_js=curve_exe_js[start_idx:end_idx+1]
        curve_exe_R=curve_exe_R[start_idx:end_idx+1]
        speed=speed[start_idx:end_idx+1]
        # speed=replace_outliers(np.array(speed))
        lam=calc_lam_cs(curve_exe)

        # plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'red',label='Motion Cmd')
        # #plot execution curve
        # ax.plot3D(curve_exe[:,0], curve_exe[:,1],curve_exe[:,2], 'green',label='Executed Motion')
        # ax.view_init(elev=40, azim=-145)
        # ax.set_title('Cartesian Interpolation using Motion Cmd')
        # ax.set_xlabel('x-axis (mm)')
        # ax.set_ylabel('y-axis (mm)')
        # ax.set_zlabel('z-axis (mm)')
        # plt.show()

        ##############################calcualte error########################################
        error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
        max_error=max(error)
        print(max_error)
        all_max_error.append(max_error)
        max_angle_error=max(angle_error)
        max_error_idx=np.argmax(error)#index of exe curve with max error
        _,max_error_curve_idx=calc_error(curve_exe[max_error_idx],curve[:,:3])  # index of original curve closest to max error point
        exe_error_vector=(curve[max_error_curve_idx,:3]-curve_exe[max_error_idx])           # shift vector
        
        ##############################plot error#####################################
        if i>iteration:
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
            plt.savefig(ilc_output+'error_speed_'+str(i)+'.png')
            plt.clf()
            ###########################find closest bp####################################
            bp_idx=np.absolute(breakpoints-max_error_curve_idx).argmin()
            ###########################plot for verification###################################
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='gray',label='original')
            ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c='red',label='execution')
            p_bp_np=np.array(p_bp[1:-1])      ###np version, avoid first and last extended points
            ax.scatter3D(p_bp_np[:,0], p_bp_np[:,1], p_bp_np[:,2], c=p_bp_np[:,2], cmap='Greens',label='breakpoints')
            ax.scatter(curve_exe[max_error_idx,0], curve_exe[max_error_idx,1], curve_exe[max_error_idx,2],c='orange',label='worst case')
            ax.view_init(elev=33, azim=-158)
            ax.set_xlabel('x-axis (mm)')
            ax.set_ylabel('y-axis (mm)')
            ax.set_zlabel('z-axis (mm)')
        
        
        ##########################################calculate gradient######################################
        ######gradient calculation related to nearest 3 points from primitive blended trajectory, not actual one
        ###restore trajectory from primitives
        curve_interp, curve_R_interp, curve_js_interp, breakpoints_blended=form_traj_from_bp(q_bp,primitives,robot)

        curve_js_blended,curve_blended,curve_R_blended=blend_js_from_primitive(curve_interp, curve_js_interp, breakpoints_blended, primitives,robot,zone=10)

        ###get closest to worst case point on blended trajectory
        _,max_error_curve_blended_idx=calc_error(curve_exe[max_error_idx],curve_blended)
        curve_blended_point=copy.deepcopy(curve_blended[max_error_curve_blended_idx]) 

        ###############get numerical gradient#####
        ###find closest 3 breakpoints
        order=np.argsort(np.abs(breakpoints_blended-max_error_curve_blended_idx))
        breakpoint_interp_2tweak_indices=order[:3]

        de_dp=ilc.get_gradient_from_model_xyz(q_bp,p_bp,breakpoints_blended,curve_blended,max_error_curve_blended_idx,robot.fwd(curve_exe_js[max_error_idx]),curve[max_error_curve_idx,:3],breakpoint_interp_2tweak_indices)
        p_bp, q_bp=ilc.update_bp_xyz(p_bp,q_bp,de_dp,max_error,breakpoint_interp_2tweak_indices)

        if i>iteration:
            for m in breakpoint_interp_2tweak_indices:
                ax.scatter(p_bp[m][0], p_bp[m][1], p_bp[m][2],c='blue',label='adjusted breakpoints')
            plt.legend()
            # plt.show()
            plt.savefig(ilc_output+'traj_3d_'+str(i)+'.png')
            plt.clf()

            # save results
            with open(ilc_output+'error_'+str(i)+'.npy','wb') as f:
                np.save(f,error)
            with open(ilc_output+'normal_error_'+str(i)+'.npy','wb') as f:
                np.save(f,angle_error)
            with open(ilc_output+'speed_'+str(i)+'.npy','wb') as f:
                np.save(f,speed)
            with open(ilc_output+'lambda_'+str(i)+'.npy','wb') as f:
                np.save(f,lam)
            with open(ilc_output+'max_error.npy','wb') as f:
                np.save(f,max_error)


if __name__ == "__main__":
    main()