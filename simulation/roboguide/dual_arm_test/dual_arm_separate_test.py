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

    # s=int(1600/2.) # mm/sec in leader frame
    s=1000 # mm/sec
    # s=16 # mm/sec in leader frame
    z=100 # CNT100
    ilc_output=data_dir+'results_'+str(s)+'_'+test_type+'/'
    Path(ilc_output).mkdir(exist_ok=True)

    breakpoints1,primitives1,p_bp1,q_bp1,_=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command1.csv')
    breakpoints2,primitives2,p_bp2,q_bp2,_=ms.extract_data_from_cmd(os.getcwd()+'/'+cmd_dir+'command2.csv')
    
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
    print("moving robot")
    logged_data1,logged_data2=ms.exec_motions_multimove_separate(robot1,robot2,primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,s1_movel,s2_movel,z,z)

if __name__ == "__main__":
    main()