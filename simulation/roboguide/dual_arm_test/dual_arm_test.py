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
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *
from blending import *

def main():
    # curve
    # data_type='blade'
    # data_type='wood'
    # data_type='blade_shift'
    # data_type='curve_line'
    data_type='curve_line_1000'

    # data and curve directory
    if data_type=='blade':
        curve_data_dir='../../../data/from_NX/'
        cmd_dir='../data/curve_blade/'
    elif data_type=='wood':
        curve_data_dir='../../../data/wood/'
        cmd_dir='../data/curve_wood/'
    elif data_type=='blade_shift':
        curve_data_dir='../../../data/blade_shift/'
        cmd_dir='../data/curve_blade_shift/'
    elif data_type=='curve_line':
        curve_data_dir='../../../data/curve_line/'
        cmd_dir='../data/curve_line/'
    elif data_type=='curve_line_1000':
        curve_data_dir='../../../data/curve_line_1000/'
        cmd_dir='../data/curve_line_1000/'

    # test_type='dual_arm'
    # test_type='dual_single_arm'
    # test_type='dual_single_arm_straight' # robot2 is multiple user defined straight line
    # test_type='dual_single_arm_straight_50' # robot2 is multiple user defined straight line
    # test_type='dual_single_arm_straight_min' # robot2 is multiple user defined straight line
    # test_type='dual_single_arm_straight_min10' # robot2 is multiple user defined straight line
    # test_type='dual_arm_10'
    test_type='dual_straight' # test: robot2 move a simple straight line

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
    elif data_type=='blade_shift':
        ms = MotionSendFANUC(robot1=robot1,robot2=robot2,utool2=4)
    elif data_type=='curve_line' or data_type=='curve_line_1000':
        ms = MotionSendFANUC(robot1=robot1,robot2=robot2,utool2=5)

    s=200 # mm/sec in leader frame
    z=100 # CNT100

    breakpoints1,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command1.csv')
    breakpoints2,primitives2,p_bp2,q_bp2=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command2.csv')

    # tp_follow = TPMotionProgram()
    # tp_lead = TPMotionProgram()
    # client = FANUCClient()
    # #### move to start
    # # robot1
    # j0 = joint2robtarget(q_bp1[0][0],robot1,1,1,2)
    # tp_follow.moveJ(j0,50,'%',-1)
    # tp_follow.moveJ(j0,5,'%',-1)
    # # robot2
    # j0 = joint2robtarget(q_bp2[0][0],robot2,2,1,2)
    # tp_lead.moveJ(j0,50,'%',-1)
    # tp_lead.moveJ(j0,5,'%',-1)
    # client.execute_motion_program_coord(tp_lead,tp_follow)
    # exit()

    ###extension
    if data_type=='wood':
        extension_d=200
    elif data_type=='blade':
        extension_d=300
    elif data_type=='blade_shift':
        extension_d=300
    elif data_type=='curve_line' or data_type=='curve_line_1000':
        extension_d=0
        # extension_d=300
    # extension_d=0
    if extension_d != 0:
        try:
            _,primitives1,p_bp1,q_bp1=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command_arm1_extend_'+str(extension_d)+'.csv')
            _,primitives2,p_bp2,q_bp2=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command_arm2_extend_'+str(extension_d)+'.csv')
            print("Extension file existed.")
            primitives1=np.array(primitives1)
            primitives2=np.array(primitives2)
            primitives1[:]='movel_fit'
            primitives2[:]='movel_fit'
        except:
            print("Extension file not existed.")
            p_bp1,q_bp1,p_bp2,q_bp2,step_to_extend_end=ms.extend_dual_relative(ms.robot1,p_bp1,q_bp1,primitives1,ms.robot2,p_bp2,q_bp2,primitives2,breakpoints1,base2_T,extension_d=extension_d)
            print(len(primitives1))
            print(len(primitives2))
            # not_coord_step=3 # after 3 bp of workpiece, no more coordination
            # coord_primitives[-(step_to_extend_end-not_coord_step):]=0
            primitives1=np.array(primitives1)
            primitives2=np.array(primitives2)
            primitives1[-int(step_to_extend_end/2):]='movej_fit'
            primitives2[-int(step_to_extend_end/2):]='movej_fit'

            df=DataFrame({'primitives':primitives1,'points':p_bp1,'q_bp':q_bp1})
            df.to_csv(cmd_dir+'command_arm1_extend_'+str(extension_d)+'.csv',header=True,index=False)
            df=DataFrame({'primitives':primitives2,'points':p_bp2,'q_bp':q_bp2})
            df.to_csv(cmd_dir+'command_arm2_extend_'+str(extension_d)+'.csv',header=True,index=False)
        
    
    coord_primitives=np.ones(len(primitives1))
    # coord_primitives=np.zeros(len(primitives1))

    ###execution with plant
    logged_data=ms.exec_motions_multimove(robot1,robot2,primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,s,z,coord_primitives)
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
    print('Max Error:',max(error),'Ave. Speed:',ave_speed,'Std. Speed:',np.std(speed),'Std/Ave (%):',np.std(speed)/ave_speed*100)
    print('Max Speed:',max(speed),'Min Speed:',np.min(speed),'Ave. Error:',np.mean(error),'Min Error:',np.min(error),"Std. Error:",np.std(error))
    print('Max Ang Error:',max(np.degrees(angle_error)),'Min Ang Error:',np.min(np.degrees(angle_error)),'Ave. Ang Error:',np.mean(np.degrees(angle_error)),"Std. Ang Error:",np.std(np.degrees(angle_error)))
    print("===========================================")

    ##############################plot error#####################################
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(lam, speed, 'g-', label='Speed')
    ax2.plot(lam, error, 'b-',label='Error')
    ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
    draw_y_max=max(speed)*1.05
    ax1.axis(ymin=0,ymax=draw_y_max)
    draw_y_max=np.max([max(error)*1.05,0.5])
    ax2.axis(ymin=0,ymax=draw_y_max)
    ax1.set_xlabel('lambda (mm)')
    ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
    ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
    plt.title("Speed and Error Plot")
    ax1.legend(loc=0)
    ax2.legend(loc=0)
    plt.legend()
    # plt.savefig('data/error_speed')
    # plt.clf()
    plt.show()

    plot_traj=False
    if plot_traj:
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(relative_path[:,0], relative_path[:,1],relative_path[:,2], 'red',label='original')
        ax.scatter3D(relative_path[breakpoints1[:-1],0], relative_path[breakpoints1[:-1],1],relative_path[breakpoints1[:-1],2], 'blue')
        ax.plot3D(relative_path_exe[:,0], relative_path_exe[:,1],relative_path_exe[:,2], 'green',label='execution')
        plt.legend()
        plt.show()

    ########################## plot joint ##########################
    plot_joint=False
    if plot_joint:
        curve_exe_js1=np.array(curve_exe_js1)
        curve_exe_js2=np.array(curve_exe_js2)
        for i in range(1,3):
            # robot
            if i==1:
                this_robot=robot1
                this_curve_js_exe=curve_exe_js1
            else:
                this_robot=robot2
                this_curve_js_exe=curve_exe_js2
            fig, ax = plt.subplots(4,6)
            dt=np.gradient(timestamp)
            for j in range(6):
                ax[0,j].plot(lam,this_curve_js_exe[:,j])
                ax[0,j].axis(ymin=this_robot.lower_limit[j]*1.05,ymax=this_robot.upper_limit[j]*1.05)
                dq=np.gradient(this_curve_js_exe[:,j])
                dqdt=dq/dt
                ax[1,j].plot(lam,dqdt)
                ax[1,j].axis(ymin=-this_robot.joint_vel_limit[j]*1.05,ymax=this_robot.joint_vel_limit[j]*1.05)
                d2qdt2=np.gradient(dqdt)/dt
                ax[2,j].plot(lam,d2qdt2)
                ax[2,j].axis(ymin=-this_robot.joint_acc_limit[j]*1.05,ymax=this_robot.joint_acc_limit[j]*1.05)
                d3qdt3=np.gradient(d2qdt2)/dt
                ax[3,j].plot(lam,d3qdt3)
                ax[3,j].axis(ymin=-this_robot.joint_jrk_limit[j]*1.05,ymax=this_robot.joint_jrk_limit[j]*1.05)
            # plt.title('Robot '+str(i)+' joint trajectoy/velocity/acceleration.')
            plt.show()
        
        # fig, ax = plt.subplots(2,3)
        # for j in range(6):
        #     dq=np.gradient(this_curve_js_exe[:,j])
        #     ax[int(j/3),j%3].plot(lam,dq/dt)
        #     ax.axis(ymin=-this_robot.joint_vel_limit*1.05,ymax=this_robot.joint_vel_limit*1.05)
        # plt.title('Robot '+str(i)+' joint velcity.')
        # plt.show()
        # fig, ax = plt.subplots(2,3)
        # for j in range(6):
        #     dq=np.gradient(this_curve_js_exe[:,j])
        #     d2qdt2=np.gradient(dq/dt)/dt
        #     ax[int(j/3),j%3].plot(lam,d2qdt2)
        #     ax.axis(ymin=-this_robot.joint_acc_limit*1.05,ymax=this_robot.joint_acc_limit*1.05)
        # plt.title('Robot '+str(i)+' joint acceleration.')
        # plt.show()


if __name__ == "__main__":
    main()