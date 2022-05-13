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
    data_dir="greedy_dual_output/"

    vmax = speeddata(10000,9999999,9999999,999999)
    v680 = speeddata(680,9999999,9999999,999999)
    speed={"v500":v500}
    zone={"z10":z10}

    for s in speed:
        for z in zone: 
            curve_exe_js=ms.exe_from_file_multimove(data_dir+"command1.csv",data_dir+"command2.csv",data_dir+"curve_fit_js1.csv",data_dir+"curve_fit_js2.csv",speed[s],speed[s],zone[z],zone[z])
   

            f = open(data_dir+"curve_exe"+"_"+s+"_"+z+".csv", "w")
            f.write(curve_exe_js)
            f.close()

if __name__ == "__main__":
    main()