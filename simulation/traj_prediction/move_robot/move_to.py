import numpy as np
from math import pi, cos, sin, radians
import time
import general_robotics_toolbox as rox
from general_robotics_toolbox import general_robotics_toolbox_invkin as roxinv

from abb_motion_program_exec_client import *

client = MotionProgramExecClient()
ENDWAITTIME = 0.1

# import sys
# sys.path.append('../../toolbox')
# from robots_def import *

# dt = 0.004
# robot = abb6640(R_tool=Ry(np.radians(90)),p_tool=np.array([0,0,0]))

SLEEPTIME=5

def send_to(p1,quat1,cf,vel,zone):
    
    robt1 = robtarget(p1, quat1, confdata(cf[0],cf[1],cf[2],cf[3]),[9E+09]*6)

    # move to the start pose and wait
    mp = MotionProgram()
    mp.MoveJ(robt1,v50,fine)
    client.execute_motion_program(mp)
    time.sleep(SLEEPTIME)
    
    return None

z_height = 1400
side_l = 200
all_Rq = rox.R2q(rox.rot([0,1,0],pi/2))
conf = [0,0,0,1]
v_profile=v50
zone=z10
ang = radians(90)/2


x_mid = 1700
y_mid = 100
p1 = np.array([x_mid,y_mid+side_l*cos(ang),z_height])
send_to(p1,all_Rq,conf,v_profile,zone)