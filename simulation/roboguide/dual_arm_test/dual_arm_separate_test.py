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

    data_dir='../ilc_fanuc/'+data_dir
    test_type='dual_arm'

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
        # ms = MotionSendFANUC(robot1=robot1,robot2=robot2,utool2=2)

    # s=int(1600/2.) # mm/sec in leader frame
    s=1000 # mm/sec
    # s=16 # mm/sec in leader frame
    z=100 # CNT100
    save_output='data/separate_controller_notsparate/'
    Path(save_output).mkdir(exist_ok=True)

    breakpoints1,primitives1,p_bp1,q_bp1,_=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command1.csv')
    breakpoints2,primitives2,p_bp2,q_bp2,_=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command2.csv')
    
    multi_peak_threshold=0.2
    save_fig=True
    ########## read data ##############
    use_iteration=27
    cmd_folder = data_dir+'results_1000_dual_arm_extend1_nospeedreg/'
    _,primitives1,p_bp1,q_bp1,_=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_folder+'command_arm1_'+str(use_iteration)+'.csv')
    _,primitives2,p_bp2,q_bp2,_=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_folder+'command_arm2_'+str(use_iteration)+'.csv')
    # read speed (for now)
    _,_,_,_,s1_movel=ms.extract_data_from_cmd(os.getcwd()+'/'+data_dir+'results_1000_dual_arm/'+'command_arm1_'+str(2)+'.csv')
    _,_,_,_,s2_movel=ms.extract_data_from_cmd(os.getcwd()+'/'+data_dir+'results_1000_dual_arm/'+'command_arm2_'+str(2)+'.csv')

    step_start1,step_start2,step_end1,step_end2=1,1,len(p_bp1)-2,len(p_bp2)-2

    ###execution with plant
    draw_speed_max=None
    draw_error_max=None
    print("moving robot")
    for i in range(5):
        # # thread base motion
        # logged_data1,logged_data2=ms.exec_motions_multimove_separate(robot1,robot2,primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,s1_movel,s2_movel,z,z)
        # connecting DI DO motion
        logged_data1,logged_data2=ms.exec_motions_multimove_separate2(robot1,robot2,primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,s1_movel,s2_movel,z,z)
        StringData=StringIO(logged_data1.decode('utf-8'))
        df1 = read_csv(StringData, sep =",")
        StringData=StringIO(logged_data2.decode('utf-8'))
        df2 = read_csv(StringData, sep =",")
        # ##############################data analysis#####################################
        lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R = ms.logged_data_analysis_multimove_connect(df1,df2,base2_R,base2_p,realrobot=False)
        
        # logged_data=ms.exec_motions_multimove_nocoord(robot1,robot2,primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,s1_movel,s2_movel,z,z)
        # StringData=StringIO(logged_data.decode('utf-8'))
        # df = read_csv(StringData, sep =",")
        ##############################data analysis#####################################
        lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R = ms.logged_data_analysis_multimove(df,base2_R,base2_p,realrobot=False)
    
        # save js
        dfj=DataFrame({'timestamp':timestamp,\
            'q11':curve_exe_js1[:,0],'q12':curve_exe_js1[:,1],'q13':curve_exe_js1[:,2],'q14':curve_exe_js1[:,3],'q15':curve_exe_js1[:,4],'q16':curve_exe_js1[:,5],\
            'q21':curve_exe_js2[:,0],'q22':curve_exe_js2[:,1],'q23':curve_exe_js2[:,2],'q24':curve_exe_js2[:,3],'q25':curve_exe_js2[:,4],'q26':curve_exe_js2[:,5]})
        dfj.to_csv(save_output+'iter_'+str(i)+'_curve_exe_js.csv',header=False,index=False)
        #############################chop extension off##################################
        lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
            ms.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])

        error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])
        #############################error peak detection###############################
        find_peak_dist = 20/(lam[int(len(lam)/2)]-lam[int(len(lam)/2)-1])
        if find_peak_dist<1:
            find_peak_dist=1
        peaks,_=find_peaks(error,height=multi_peak_threshold,prominence=0.05,distance=find_peak_dist)		###only push down peaks higher than height, distance between each peak is 20mm, threshold to filter noisy peaks
        if len(peaks)==0 or np.argmax(error) not in peaks:
            peaks=np.append(peaks,np.argmax(error))\
        
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
            plt.savefig(save_output+'iteration_'+str(i))
            plt.clf()
        else:
            plt.show()

if __name__ == "__main__":
    main()