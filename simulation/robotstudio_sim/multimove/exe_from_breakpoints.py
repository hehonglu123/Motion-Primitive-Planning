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
    ms = MotionSend(robot1=abb1200(d=50),robot2=abb6640(),tool1=tooldata(True,pose([75,0,493.30127019],[quatR[0],quatR[1],quatR[2],quatR[3]]),loaddata(1,[0,0,0.001],[1,0,0,0],0,0,0)),tool2=tooldata(True,pose([50,0,450],[quatR[0],quatR[1],quatR[2],quatR[3]]),loaddata(1,[0,0,0.001],[1,0,0,0],0,0,0)))
    data_dir="from_NX/greedy_dual_output/"

    vmax = speeddata(10000,9999999,9999999,999999)
    v680 = speeddata(680,9999999,9999999,999999)
    speed={'v1000':v1000}
    # speed={'v50':v50,'v100':v100,'v200':v200,'v400':v400,'v500':v500,'vmax':vmax}
    zone={'z10':z10}

    for s in speed:
        for z in zone: 
            curve_exe_js=ms.exe_from_file_multimove(data_dir+"command1.csv",data_dir+"command2.csv",data_dir+"curve_fit_js1.csv",data_dir+"curve_fit_js2.csv",speed[s],speed[s],zone[z],zone[z])
   

            f = open(data_dir+"curve_exe"+"_"+s+"_"+z+".csv", "w")
            f.write(curve_exe_js)
            f.close()

if __name__ == "__main__":
    main()