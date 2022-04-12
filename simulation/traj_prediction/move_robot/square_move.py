import numpy as np
from math import pi, cos, sin, radians
import time
import general_robotics_toolbox as rox
from general_robotics_toolbox import general_robotics_toolbox_invkin as roxinv

from abb_motion_program_exec_client import *

# client = MotionProgramExecClient(base_url='http://192.168.55.1:80')
client = MotionProgramExecClient()
ENDWAITTIME = 0.1

# import sys
# sys.path.append('../../toolbox')
# from robots_def import *

# dt = 0.004
# robot = abb6640(R_tool=Ry(np.radians(90)),p_tool=np.array([0,0,0]))

SLEEPTIME=0.1

def send_square(p1,p2,p3,p4,quat1,quat2,quat3,quat4,cf,vel,zone):
    
    robt1 = robtarget(p1, quat1, confdata(cf[0],cf[1],cf[2],cf[3]),[9E+09]*6)
    robt2 = robtarget(p2, quat2, confdata(cf[0],cf[1],cf[2],cf[3]),[9E+09]*6) 
    robt3 = robtarget(p3, quat3, confdata(cf[0],cf[1],cf[2],cf[3]),[9E+09]*6)
    robt4 = robtarget(p4, quat4, confdata(cf[0],cf[1],cf[2],cf[3]),[9E+09]*6) 

    # move to the start pose and wait
    mp = MotionProgram()
    mp.MoveJ(robt4,v50,fine)
    client.execute_motion_program(mp)
    time.sleep(SLEEPTIME)

    # the trajectory
    mp = MotionProgram()
    mp.MoveL(robt1,vel,zone)
    mp.MoveL(robt2,vel,zone)
    mp.MoveL(robt3,vel,zone)
    mp.MoveL(robt4,vel,fine)
    mp.WaitTime(ENDWAITTIME)

    # print(mp.get_program_rapid())
    log_results = client.execute_motion_program(mp)
    time.sleep(SLEEPTIME)
    
    return log_results

z_height = 1400
side_l = 200
all_Rq = rox.R2q(rox.rot([0,1,0],pi/2))
conf = [0,0,0,1]
v_profile=v50
zone=z10
ang = radians(90)/2

# left square
x_mid = 1700
y_mid = 100
p4 = np.array([x_mid,y_mid+side_l*cos(ang),z_height])
p1 = np.array([x_mid+side_l*cos(ang),y_mid,z_height])
p2 = np.array([x_mid,y_mid-side_l*cos(ang),z_height])
p3 = np.array([x_mid-side_l*cos(ang),y_mid,z_height])

log = send_square(p1,p2,p3,p4,all_Rq,all_Rq,all_Rq,all_Rq,conf,v_profile,zone)

with open("square_move_v50_left.csv","wb") as f:
    f.write(log)

# middle square
x_mid = 1700
y_mid = 0
p4 = np.array([x_mid,y_mid+side_l*cos(ang),z_height])
p1 = np.array([x_mid+side_l*cos(ang),y_mid,z_height])
p2 = np.array([x_mid,y_mid-side_l*cos(ang),z_height])
p3 = np.array([x_mid-side_l*cos(ang),y_mid,z_height])

log = send_square(p1,p2,p3,p4,all_Rq,all_Rq,all_Rq,all_Rq,conf,v_profile,zone)

with open("square_move_v50_middle.csv","wb") as f:
    f.write(log)

# right square
x_mid = 1700
y_mid = -100
p4 = np.array([x_mid,y_mid+side_l*cos(ang),z_height])
p1 = np.array([x_mid+side_l*cos(ang),y_mid,z_height])
p2 = np.array([x_mid,y_mid-side_l*cos(ang),z_height])
p3 = np.array([x_mid-side_l*cos(ang),y_mid,z_height])

log = send_square(p1,p2,p3,p4,all_Rq,all_Rq,all_Rq,all_Rq,conf,v_profile,zone)

with open("square_move_v50_right.csv","wb") as f:
    f.write(log)

# velocity == v500
v_profile=v500

# left square
x_mid = 1700
y_mid = 100
p4 = np.array([x_mid,y_mid+side_l*cos(ang),z_height])
p1 = np.array([x_mid+side_l*cos(ang),y_mid,z_height])
p2 = np.array([x_mid,y_mid-side_l*cos(ang),z_height])
p3 = np.array([x_mid-side_l*cos(ang),y_mid,z_height])

log = send_square(p1,p2,p3,p4,all_Rq,all_Rq,all_Rq,all_Rq,conf,v_profile,zone)

with open("square_move_v500_left.csv","wb") as f:
    f.write(log)

# middle square
x_mid = 1700
y_mid = 0
p4 = np.array([x_mid,y_mid+side_l*cos(ang),z_height])
p1 = np.array([x_mid+side_l*cos(ang),y_mid,z_height])
p2 = np.array([x_mid,y_mid-side_l*cos(ang),z_height])
p3 = np.array([x_mid-side_l*cos(ang),y_mid,z_height])

log = send_square(p1,p2,p3,p4,all_Rq,all_Rq,all_Rq,all_Rq,conf,v_profile,zone)

with open("square_move_v500_middle.csv","wb") as f:
    f.write(log)

# right square
x_mid = 1700
y_mid = -100
p4 = np.array([x_mid,y_mid+side_l*cos(ang),z_height])
p1 = np.array([x_mid+side_l*cos(ang),y_mid,z_height])
p2 = np.array([x_mid,y_mid-side_l*cos(ang),z_height])
p3 = np.array([x_mid-side_l*cos(ang),y_mid,z_height])

log = send_square(p1,p2,p3,p4,all_Rq,all_Rq,all_Rq,all_Rq,conf,v_profile,zone)

with open("square_move_v500_right.csv","wb") as f:
    f.write(log)
