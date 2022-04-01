import numpy as np
from math import pi, cos, sin, radians
import time
import general_robotics_toolbox as rox
from general_robotics_toolbox import general_robotics_toolbox_invkin as roxinv

from abb_motion_program_exec_client import *

client = MotionProgramExecClient()
ENDWAITTIME = 0.1

def send_movels(p1,q1,cf1,p2,q2,cf2,p3,q3,cf3,vel,zone):
    
    robt1 = robtarget(p1, q1, confdata(cf1[0],cf1[1],cf1[2],cf1[3]),[9E+09]*6)
    robt2 = robtarget(p2, q2, confdata(cf2[0],cf2[1],cf2[2],cf2[3]),[9E+09]*6) 
    robt3 = robtarget(p3, q3, confdata(cf3[0],cf3[1],cf3[2],cf3[3]),[9E+09]*6) 

    # move to the start pose and wait
    mp = MotionProgram()
    mp.MoveJ(robt1,v500,fine)
    client.execute_motion_program(mp)
    time.sleep(0.1)

    # the trajectory
    mp = MotionProgram()
    mp.MoveL(robt2,vel,zone)
    mp.MoveL(robt3,vel,fine)
    mp.WaitTime(ENDWAITTIME)
    log_results = client.execute_motion_program(mp)
    
    return log_results

def send_to_pose(p1,q1,cf1):

    robt = robtarget(p1, q1, confdata(cf1[0],cf1[1],cf1[2],cf1[3]),[9E+09]*6)
    mp = MotionProgram()
    mp.MoveJ(robt,v500,fine)
    mp.WaitTime(ENDWAITTIME)
    log_results = client.execute_motion_program(mp)

    return log_results


# configuration of the test
start_p = np.array([2300,1000,600])
end_p = np.array([1300,-1000,600])
all_Rq = rox.R2q(rox.rot([0,1,0],pi/2))
conf = [0,0,0,1]
side_l = 200
x_divided = 11
step_x = (end_p[0]-start_p[0])/(x_divided-1)
y_divided = 11
step_y = (end_p[1]-start_p[1])/(y_divided-1)
all_vel = [v50,v500]
angels = [90,120,150]

# send_to_pose(start_p,all_Rq,conf)
# send_to_pose(end_p,all_Rq,conf)
p = end_p
# p[0] = end_p[0]
# p[1] = start_p[1]+step_y*5
send_to_pose(p,all_Rq,conf)