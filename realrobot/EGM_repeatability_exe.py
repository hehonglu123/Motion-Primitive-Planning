########
# This module utilized https://github.com/johnwason/abb_motion_program_exec
# and send whatever the motion primitives that algorithms generate
# to RobotStudio
########

import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from abb_motion_program_exec_client import *
from robots_def import *
from error_check import *
from MotionSend import *
from io import StringIO
from lambda_calc import *
from EGM_toolbox import *

def main():
    robot=abb6640(d=50)

    dataset="../data/wood/"
    data_dir=dataset+'baseline/100L/'

    curve = read_csv(dataset+"Curve_in_base_frame.csv",header=None).values
    curve_js=read_csv(dataset+"Curve_js.csv",header=None).values
    

    egm = rpi_abb_irc5.EGM(port=6510)

    et=EGM_toolbox(egm,robot)

    extension_start=100
    extension_end=0
    

    speed=[100,200,300,400]
    iterations=5
    for v in speed:
        for i in range(iterations):
            curve_cmd_js=curve_js[et.downsample24ms(curve,v)]

            curve_cmd_js_ext=et.add_extension_egm_js(curve_cmd_js,extension_start=extension_start,extension_end=extension_end)

            et.jog_joint(curve_cmd_js_ext[0])
            timestamp,curve_exe_js=et.traverse_curve_js(curve_cmd_js_ext)

            time.sleep(1)
            df=DataFrame({'timestampe':timestamp,'q0':curve_exe_js[:,0],'q1':curve_exe_js[:,1],'q2':curve_exe_js[:,2],'q3':curve_exe_js[:,3],'q4':curve_exe_js[:,4],'q5':curve_exe_js[:,5]})
            df.to_csv('recorded_data/repeatibility_EGM/v'+str(v)+'_iteration'+str(i)+'.csv',header=False,index=False)

if __name__ == "__main__":
    main()