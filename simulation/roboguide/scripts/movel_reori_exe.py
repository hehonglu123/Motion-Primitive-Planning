from math import radians,degrees
import numpy as np
from pandas import read_csv,DataFrame

from general_robotics_toolbox import *
import sys

# from simulation.roboguide.fanuc_toolbox.fanuc_client import FANUCClient, TPMotionProgram, joint2robtarget, jointtarget, robtarget
# from toolbox.robots_def import arb_robot, m900ia
sys.path.append('../../../toolbox')
from robots_def import *
from lambda_calc import *
from error_check import *
sys.path.append('../../../circular_fit')
from toolbox_circular_fit import *
sys.path.append('../fanuc_toolbox')
from fanuc_client import *

# generate curve with linear orientation
# robot=abb6640(d=50)
robot=m900ia(d=50)
###generate a continuous arc, with linear orientation
start_p = np.array([2200,500, 1000])
end_p = np.array([2200, -500, 1000])

# curve=arc_from_3point(start_p,end_p,mid_p,N=5001)
curve=np.linspace(start_p,end_p,50001)
#get orientation
R_init=Ry(np.radians(135))
R_end=Ry(np.radians(90))
# print(rox.R2rot(R_init))
# print(rox.R2rot(R_end))
# print(degrees(rox.R2rot(R_init)[1]))
# print(np.rad2deg(robot.inv(start_p,R_init)))
# print(np.rad2deg(robot.inv(end_p,R_end)))

# q_init=robot.inv(start_p,R_init)[0]
q_init=robot.inv(start_p,R_init)[1]

#interpolate orientation and solve inv kin
curve_js=[q_init]
R_all=[R_init]
k,theta=R2rot(np.dot(R_end,R_init.T))
# theta=np.pi/4 #force 45deg change
for i in range(1,len(curve)):
    angle=theta*i/(len(curve)-1)
    R_temp=rot(k,angle)
    R_all.append(np.dot(R_temp,R_init))
    q_all=np.array(robot.inv(curve[i],R_all[-1]))
    ###choose inv_kin closest to previous joints
    temp_q=q_all-curve_js[-1]
    order=np.argsort(np.linalg.norm(temp_q,axis=1))
    curve_js.append(q_all[order[0]])

curve_js=np.array(curve_js)
R_all=np.array(R_all)
print('total length: ',calc_lam_cs(curve)[-1])

# save directory
save_dir='../data/movel_reori/'

DataFrame(curve_js).to_csv(save_dir+'Curve_js.csv',header=False,index=False)

# exit()

# define speed and zone (CNT)
# all_speed = [50,200]
all_speed = [500,1000]

# fanuc client
client = FANUCClient()
utool_num = 2

for speed in all_speed:

    # tp program
    # move to start
    tp_pre = TPMotionProgram()
    j0 = jointtarget(1,1,utool_num,np.degrees(curve_js[0]),[0]*6)
    tp_pre.moveJ(j0,50,'%',-1)
    client.execute_motion_program(tp_pre)

    # start moving along the curve: moveL
    tp = TPMotionProgram()
    robt = joint2robtarget(curve_js[-1],robot,1,1,utool_num)
    tp.moveL(robt,speed,'mmsec',-1)
    
    # execute 
    res = client.execute_motion_program(tp)
    # Write log csv to file
    with open(save_dir+"movel_"+str(speed)+".csv","wb") as f:
        f.write(res)