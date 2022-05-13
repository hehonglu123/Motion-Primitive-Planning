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
    #determine correct commanded speed to keep error within 1mm
    thresholds=[0.1,0.2,0.5,0.9]
    for threshold in thresholds:
        ms = MotionSend()
        # data_dir="fitting_output_new/python_qp_movel/"
        data_dir="../../../data/from_NX/baseline/0.1/"

        vmax = speeddata(10000,9999999,9999999,999999)
        v680 = speeddata(680,9999999,9999999,999999)
        speed={"vmax":vmax}
        zone={"z10":z10}

        for s in speed:
            for z in zone: 
                curve_exe_js=ms.exe_from_file(data_dir+"command.csv",data_dir+"curve_fit_js.csv",speed[s],zone[z])
       

                f = open(data_dir+"curve_exe"+"_"+s+"_"+z+".csv", "w")
                f.write(curve_exe_js)
                f.close()

if __name__ == "__main__":
    main()