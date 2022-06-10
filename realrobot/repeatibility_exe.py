########
# This module utilized https://github.com/johnwason/abb_motion_program_exec
# and send whatever the motion primitives that algorithms generate
# to RobotStudio
########

import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from io import StringIO
from lambda_calc import *

def main():
    robot=abb6640(d=50)

    dataset="../data/from_NX/"
    data_dir=dataset+'baseline/100L/'

    curve = read_csv(dataset+"Curve_in_base_frame.csv",header=None).values

    speed=[200,400,600,800]
    iterations=5
    for v in speed:
        for i in range(iterations):
            ms = MotionSend(url='http://192.168.55.1:80')
            
            ###update velocity profile
            v_cmd = speeddata(v,9999999,9999999,999999)

            ###read data
            breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd(data_dir+'command.csv')
            ###extension
            primitives,p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp)

            logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,v_cmd,z10)


            # Write log csv to file
            with open("recorded_data/repeatibility/v"+str(v)+'_iteration'+str(i)+".csv","w") as f:
                f.write(logged_data)

if __name__ == "__main__":
    main()