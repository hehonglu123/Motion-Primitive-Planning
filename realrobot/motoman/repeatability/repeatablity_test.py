import numpy as np
from general_robotics_toolbox import *
import sys
sys.path.append('../../toolbox')
from robots_def import *
from error_check import *
from realrobot import *
from MotionSend_motoman import *
from lambda_calc import *

def main():
    robot=robot_obj('MA2010_A0',def_path='../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun2.csv',\
    pulse2deg_file_path='../../config/MA2010_A0_pulse2deg.csv',d=50)
    ms = MotionSend(robot)

    dataset='curve_1/'
    solution_dir='baseline_motoman/'
    data_dir='../../data/'+dataset+solution_dir
    cmd_dir=data_dir+'100L/'

    curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
    breakpoints,primitives, p_bp,q_bp=ms.extract_data_from_cmd(cmd_dir+"command.csv")
    p_bp,q_bp,primitives,breakpoints=ms.extend2(robot,q_bp,primitives,breakpoints,p_bp)

   
    error_threshold=0.5
    angle_threshold=np.radians(3)

    speed=[100,150,200,250,300]
    num_runs=[4,5,6]

    for v in speed:
        for N in num_runs:
            curve_js_all_new, avg_curve_js, timestamp_d=average_N_exe(ms,robot,primitives,breakpoints,p_bp,q_bp,v,z,curve,'recorded_data/curve_1/v%d_N%d/' % (v,N),N=N)