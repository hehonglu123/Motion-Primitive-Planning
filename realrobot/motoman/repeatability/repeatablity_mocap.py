import numpy as np
from general_robotics_toolbox import *
import sys
from RobotRaconteur.Client import *

sys.path.append('../../../toolbox')
from MocapPoseListener import *
from robot_def import *
from error_check import *
from realrobot import *
from MotionSend_motoman import *
from lambda_calc import *

def main():
    config_dir='../../../config/'
    robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun2.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=50,  base_marker_config_file=config_dir+'MA2010_marker_config.yaml',\
        tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')
    # add d
    d=50-15
    T_d1_d2 = Transform(np.eye(3),p=[0,0,d])
    robot.T_tool_toolmarker = robot.T_tool_toolmarker*T_d1_d2

    mocap_url = 'rr+tcp://192.168.55.10:59823?service=optitrack_mocap'
    mocap_url = mocap_url
    mocap_cli = RRN.ConnectService(mocap_url)

    mpl_obj = MocapPoseListener(mocap_cli,[robot],collect_base_stop=1,use_static_base=True)


    ms = MotionSend(robot)

    dataset='curve_1/'
    solution_dir='baseline_motoman/'
    data_dir='../../../data/'+dataset+solution_dir
    cmd_dir=data_dir+'100L/'

    curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
    breakpoints,primitives, p_bp,q_bp=ms.extract_data_from_cmd(cmd_dir+"command.csv")
    p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp,extension_start=50,extension_end=50)

   
    error_threshold=0.5
    angle_threshold=np.radians(3)


    speed=[200]
    num_runs=[2]
    z=8
    for N in num_runs:
        for v in speed:  
            curve_exe, curve_exe_R, timestamp=average_N_exe_mocap(mpl_obj,ms,robot,primitives,breakpoints,p_bp,q_bp,v,z,curve,'recorded_data/'+dataset+'v%d_N%d/' % (v,N),N=N)


if __name__ == '__main__':
    main()