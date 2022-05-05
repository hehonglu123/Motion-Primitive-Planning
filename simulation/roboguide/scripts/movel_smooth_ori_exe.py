from math import radians,degrees,cos,sin
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
mid_p=(start_p+end_p)/2

# save directory
save_dir='../data/movel_smooth_ori/'

# define speed and zone (CNT)
all_speed = [50,200,500,1000]
all_zone = [25,50,75,100]
all_angle = [0,1,5,10,30]

# fanuc client
client = FANUCClient()
utool_num = 2

for ang in all_angle:

    lam_f=np.linalg.norm(end_p-start_p)
    lam=np.linspace(0,lam_f,num=50001)

    #generate linear segment
    a1,b1,c1=lineFromPoints([lam[0],start_p[0]],[lam[-1],end_p[0]])
    a2,b2,c2=lineFromPoints([lam[0],start_p[1]],[lam[-1],end_p[1]])
    a3,b3,c3=lineFromPoints([lam[0],start_p[2]],[lam[-1],end_p[2]])
    curve=np.vstack(((-a1*lam-c1)/b1,(-a2*lam-c2)/b2,(-a3*lam-c3)/b3)).T

    ###start rotation by 'ang' deg in orientation vector
    R_init=Ry(np.radians(135))
    R_end=Ry(np.radians(90))
    # q_init=robot.inv(start_p,R_init)[0]
    q_init=robot.inv(start_p,R_init)[1]

    #interpolate orientation and solve inv kin
    curve_js=[q_init]
    R_all=[R_init]
    k,theta=R2rot(np.dot(R_end,R_init.T))
    k2=np.array([-sin(radians(ang)),-cos(radians(ang)),0])

    bp_idx=int((len(curve)+1)/2)
    ###first half orientation
    for i in range(1,bp_idx):
        angle=theta*i/(len(curve)-1)
        R_temp=rot(k,angle)
        R_all.append(np.dot(R_temp,R_init))
        
        q_all=np.array(robot.inv(curve[i],R_all[-1]))
        ###choose inv_kin closest to previous joints
        temp_q=q_all-curve_js[-1]
        order=np.argsort(np.linalg.norm(temp_q,axis=1))
        curve_js.append(q_all[order[0]])
    ###second half orientation
    for i in range(bp_idx,len(curve)):
        angle=(theta/2)*(i-bp_idx+1)/(len(curve)-bp_idx)
        R_temp=rot(k2,angle)
        R_all.append(np.dot(R_temp,R_all[bp_idx-1]))
        
        q_all=np.array(robot.inv(curve[i],R_all[-1]))
        ###choose inv_kin closest to previous joints
        temp_q=q_all-curve_js[-1]
        order=np.argsort(np.linalg.norm(temp_q,axis=1))
        curve_js.append(q_all[order[0]])

    curve_js=np.array(curve_js)
    R_all=np.array(R_all)
    print('total length: ',calc_lam_cs(curve)[-1])
    DataFrame(curve_js).to_csv(save_dir+'Curve_js_'+str(ang)+'.csv',header=False,index=False)

    continue

    for speed in all_speed:
        for zone in all_zone:

            # tp program
            # move to start
            tp_pre = TPMotionProgram()
            j0 = jointtarget(1,1,utool_num,np.degrees(curve_js[0]),[0]*6)
            tp_pre.moveJ(j0,50,'%',-1)
            client.execute_motion_program(tp_pre)

            # start moving along the curve: moveL
            tp = TPMotionProgram()
            robt_mid = joint2robtarget(curve_js[bp_idx],robot,1,1,utool_num)
            tp.moveL(robt_mid,speed,'mmsec',zone)
            robt_end = joint2robtarget(curve_js[-1],robot,1,1,utool_num)
            tp.moveL(robt_end,speed,'mmsec',-1)
            
            # execute 
            res = client.execute_motion_program(tp)
            # Write log csv to file
            with open(save_dir+"movel_"+str(ang)+'_'+str(speed)+'_CNT'+str(zone)+".csv","wb") as f:
                f.write(res)