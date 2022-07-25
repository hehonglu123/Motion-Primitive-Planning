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
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *

def main():
    ms = MotionSend()
    dataset='from_NX/'
    cmd_dir='../../../data/'+dataset+'baseline/100L/'
    data_dir='fitting_output/'+dataset+'100L/'

    robot=abb6640(d=50)

    v250 = speeddata(250,9999999,9999999,999999)
    v350 = speeddata(350,9999999,9999999,999999)
    v450 = speeddata(450,9999999,9999999,999999)
    v900 = speeddata(900,9999999,9999999,999999)
    v1100 = speeddata(1100,9999999,9999999,999999)
    v1200 = speeddata(1200,9999999,9999999,999999)
    v1300 = speeddata(1300,9999999,9999999,999999)
    v1400 = speeddata(1400,9999999,9999999,999999)
    # speed={'v200':v200,'v250':v250,'v300':v300,'v350':v350,'v400':v400,'v450':v450,'v500':v500}
    speed={'v800':v800,'v900':v900,'v1000':v1000,'v1100':v1100,'v1200':v1200,'v1300':v1300,'v1400':v1400}
    zone={'z10':z10}

    for s in speed:
        for z in zone: 
            breakpoints,primitives, p_bp,q_bp=ms.extract_data_from_cmd(cmd_dir+"command.csv")
            p_bp, q_bp = ms.extend(robot, q_bp, primitives, breakpoints, p_bp)


            logged_data= ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,speed[s],zone[z])


            f = open(data_dir+"curve_exe"+"_"+s+"_"+z+".csv", "w")
            f.write(logged_data)
            f.close()

if __name__ == "__main__":
    main()