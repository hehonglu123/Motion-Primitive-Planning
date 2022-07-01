########
# This module utilized https://github.com/johnwason/abb_motion_program_exec
# and send whatever the motion primitives that algorithms generate
# to RobotStudio
########

import numpy as np
from scipy.signal import find_peaks
from general_robotics_toolbox import *
from pandas import read_csv,DataFrame
import sys
from io import StringIO
from matplotlib import pyplot as plt
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
    # data_dir="fitting_output_new/python_qp_movel/"
    # dataset='from_NX/'
    dataset='kink_30/'
    data_dir="data/"+dataset
    fitting_output='data/'+dataset
    

    # curve_js=read_csv(data_dir+'Curve_js.csv',header=None).values
    # curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
    with open('../Curve_js.npy','rb') as f:
        curve_js = np.load(f)
    with open('../Curve_in_base_frame.npy','rb') as f:
        curve = np.load(f)
    with open('../Curve_R_in_base_frame.npy','rb') as f:
        R_all = np.load(f)
        curve_normal=R_all[:,:,-1]
    curve = np.hstack((curve,curve_normal))
    
    multi_peak_threshold=0.2
    # robot=m710ic(d=50)
    robot=m900ia(d=50)

    s = 41
    z = 100
    ilc_output=fitting_output+'results_'+str(s)+'/'

    # curve_fit_js=read_csv(fitting_output+'curve_fit_js.csv',header=None).values
    # ms = MotionSend()
    # breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd(fitting_output+'command.csv')
    ###extension
    # primitives,p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp)

    try:
        breakpoints,primitives,p_bp,q_bp=extract_data_from_cmd(fitting_output+'command.csv')
    except:
        print("Convert bp to command")
        # curve_js_plan = read_csv(fitting_output+'bp_plan.csv',header=None).values
        # curve_js_plan=np.array(curve_js_plan).astype(float)
        # curve_js_plan=np.reshape(curve_js_plan,(int(len(curve_js_plan)/6),6))

        total_seg = 10
        step=int((len(curve_js)-1)/total_seg)
        breakpoints = [0]
        primitives = ['movej_fit']
        q_bp = [[curve_js[0]]]
        p_bp = [robot.fwd(curve_js[0]).p]
        for i in range(step,len(curve_js),step):
            breakpoints.append(i)
            primitives.append('movel_fit')
            q_bp.append([curve_js[i]])
            p_bp.append(robot.fwd(curve_js[i]).p)
        
        df=DataFrame({'breakpoints':breakpoints,'primitives':primitives,'points':p_bp,'q_bp':q_bp})
        df.to_csv(fitting_output+'command.csv',header=True,index=False)

    primitives,p_bp,q_bp=extend_start_end(robot,q_bp,primitives,breakpoints,p_bp,extension_d=0)

    # print(p_bp)
    # print(q_bp)
    # print(primitives)
    # print(breakpoints)
    # exit()

    ###ilc toolbox def
    ilc=ilc_toolbox(robot,primitives)

    ###TODO: extension fix start point, moveC support
    ms = MotionSendFANUC()
    iteration=20
    draw_y_max=None
    for i in range(iteration):
        
        ###execute,curve_fit_js only used for orientation
        logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,s,z)

        # print(logged_data)
        StringData=StringIO(logged_data.decode('utf-8'))
        # df = read_csv(StringData, sep =",")
        df = read_csv(StringData)
        ##############################data analysis#####################################
        lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df)

        #############################chop extension off##################################
        lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[:,:3],curve[:,3:])

        ##############################calcualte error########################################
        error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
        print('Iteration:',i,', Max Error:',max(error))
        #############################error peak detection###############################
        peaks,_=find_peaks(error,height=multi_peak_threshold,prominence=0.05,distance=20/(lam[int(len(lam)/2)]-lam[int(len(lam)/2)-1]))		###only push down peaks higher than height, distance between each peak is 20mm, threshold to filter noisy peaks

        if len(peaks)==0 or np.argmax(error) not in peaks:
            peaks=np.append(peaks,np.argmax(error))
        
        ##############################plot error#####################################
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(lam, speed, 'g-', label='Speed')
        ax2.plot(lam, error, 'b-',label='Error')
        ax2.scatter(lam[peaks],error[peaks],label='peaks')
        ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
        if draw_y_max is None:
            draw_y_max=max(error)*1.05
        ax2.axis(ymin=0,ymax=draw_y_max)

        ax1.set_xlabel('lambda (mm)')
        ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
        ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
        plt.title("Speed and Error Plot")
        ax1.legend(loc=0)

        ax2.legend(loc=0)

        # save fig
        plt.legend()
        plt.savefig(ilc_output+'iteration_'+str(i))
        plt.clf()
        # plt.show()
        # save bp
        q_bp_save = []
        for q in q_bp:
            q_bp_save.append(q[0])
        q_bp_save=np.array(q_bp_save)
        # q_bp_save=np.reshape(q_bp_save,(1,-1))
        # np.savetxt(ilc_output+"bp_grad_"+str(s)+"_iter_"+str(i)+".csv",q_bp_save,delimiter="\n")
        with open(ilc_output+"bp_grad_"+str(s)+"_iter_"+str(i)+".npy",'wb') as f:
            np.save(f,q_bp_save)
        
        
        ##########################################calculate gradient######################################
        ######gradient calculation related to nearest 3 points from primitive blended trajectory, not actual one
        ###restore trajectory from primitives
        curve_interp, curve_R_interp, curve_js_interp, breakpoints_blended=form_traj_from_bp(q_bp,primitives,robot)

        # curve_js_blended,curve_blended,curve_R_blended=blend_js_from_primitive(curve_interp, curve_js_interp, breakpoints_blended, primitives,robot,zone=10)
        # fanuc blend in cart space
        curve_js_blended,curve_blended,curve_R_blended=blend_cart_from_primitive(curve_interp,curve_R_interp,curve_js_interp,breakpoints_blended,primitives,robot,s)

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

            p_bp_dummy = []
            for j in range(len(p_bp)):
                p_bp_dummy.append([p_bp[j]])
            de_dp=ilc.get_gradient_from_model_xyz(p_bp_dummy,q_bp,breakpoints_blended,curve_blended,peak_error_curve_blended_idx,robot.fwd(curve_exe_js[peak]),curve[peak_error_curve_idx,:3],breakpoint_interp_2tweak_indices)
            
            alpha=0.2
            p_bp_dummy, q_bp=ilc.update_bp_xyz(p_bp_dummy,q_bp,de_dp,error[peak],breakpoint_interp_2tweak_indices,alpha=alpha)

            # backtracking line search
            # zho=0.9
            # termination=0.01
            # while True:
            #     p_bp_dummy_search, q_bp_search=ilc.update_bp_xyz(p_bp_dummy,q_bp,de_dp,error[peak],breakpoint_interp_2tweak_indices,alpha=alpha)
            #     curve_interp_search, curve_R_interp_search, curve_js_interp_search, breakpoints_blended_search=form_traj_from_bp(q_bp_search,primitives,robot)
            #     curve_js_blended_search,curve_blended_search,curve_R_blended_search=blend_cart_from_primitive(curve_interp_search,curve_R_interp_search,curve_js_interp_search,breakpoints_blended_search,primitives,robot,s)

            #     backtrack_error=calc_all_error(curve_blended_search,curve[:,:3])
            #     if max(backtrack_error) < error[peak]:
            #         # find alpha that decrease the objective
            #         print("Find alpha:",alpha)
            #         break

            #     alpha = zho*alpha
            #     if alpha < termination:
            #         print('alpha',alpha,'smaller than',termination,'. Use Alpha')
            #         break
            # p_bp_dummy=copy.deepcopy(p_bp_dummy_search)
            # q_bp=copy.deepcopy(q_bp_search)

            for j in range(len(p_bp)):
                p_bp[j]=p_bp_dummy[j][0]

if __name__ == "__main__":
    main()