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
mid_p=(start_p+end_p)/2

# save directory
save_dir='../data/movel_smooth_car/'

# define speed and zone (CNT)
# all_speed = [50,200,500,1000]
all_speed = [500,1000]
all_zone = [25,50,75,100]
all_angle = [0,1,5,10,30]

# fanuc client
client = FANUCClient()
utool_num = 2

for ang in all_angle:

    ###start rotation by 'ang' deg
    k=np.cross(end_p+np.array([0.1,0,0])-mid_p,start_p-mid_p)
    k=k/np.linalg.norm(k)
    theta=np.radians(ang)

    R=rot(k,theta)
    new_vec=R@(end_p-mid_p)
    end_p=mid_p+new_vec

    ###calculate lambda
    lam1_f=np.linalg.norm(mid_p-start_p)
    lam1=np.linspace(0,lam1_f,num=25001)
    lam_f=lam1_f+np.linalg.norm(mid_p-end_p)
    lam2=np.linspace(lam1_f,lam_f,num=25001)

    lam=np.hstack((lam1,lam2[1:]))

    #generate linear segment
    a1,b1,c1=lineFromPoints([lam1[0],start_p[0]],[lam1[-1],mid_p[0]])
    a2,b2,c2=lineFromPoints([lam1[0],start_p[1]],[lam1[-1],mid_p[1]])
    a3,b3,c3=lineFromPoints([lam1[0],start_p[2]],[lam1[-1],mid_p[2]])
    line1=np.vstack(((-a1*lam1-c1)/b1,(-a2*lam1-c2)/b2,(-a3*lam1-c3)/b3)).T

    a1,b1,c1=lineFromPoints([lam2[0],mid_p[0]],[lam2[-1],end_p[0]])
    a2,b2,c2=lineFromPoints([lam2[0],mid_p[1]],[lam2[-1],end_p[1]])
    a3,b3,c3=lineFromPoints([lam2[0],mid_p[2]],[lam2[-1],end_p[2]])
    line2=np.vstack(((-a1*lam2-c1)/b1,(-a2*lam2-c2)/b2,(-a3*lam2-c3)/b3)).T

    curve=np.vstack((line1,line2[1:]))

    R_init=Ry(np.radians(135))
    R_end=Ry(np.radians(90))
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
            robt_mid = joint2robtarget(curve_js[int((len(curve)+1)/2)],robot,1,1,utool_num)
            tp.moveL(robt_mid,speed,'mmsec',zone)
            robt_end = joint2robtarget(curve_js[-1],robot,1,1,utool_num)
            tp.moveL(robt_end,speed,'mmsec',-1)
            
            # execute 
            res = client.execute_motion_program(tp)
            # Write log csv to file
            with open(save_dir+"movel_"+str(ang)+'_'+str(speed)+'_CNT'+str(zone)+".csv","wb") as f:
                f.write(res)