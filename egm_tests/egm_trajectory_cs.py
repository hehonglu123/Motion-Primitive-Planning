#!/usr/bin/env python

import rpi_abb_irc5
import time
import math, sys, traceback
from pandas import *
import numpy as np
sys.path.append('../toolbox')
from robot_def import *

def direction2R(v_norm,v_tang):
    v_norm=v_norm/np.linalg.norm(v_norm)
    theta1 = np.arccos(np.dot(np.array([0,0,1]),v_norm))
    ###rotation to align z axis with curve normal
    axis_temp=np.cross(np.array([0,0,1]),v_norm)
    R1=rot(axis_temp/np.linalg.norm(axis_temp),theta1)

    ###find correct x direction
    v_temp=v_tang-v_norm * np.dot(v_tang, v_norm) / np.linalg.norm(v_norm)

    ###get as ngle to rotate
    theta2 = np.arccos(np.dot(R1[:,0],v_temp/np.linalg.norm(v_temp)))


    axis_temp=np.cross(R1[:,0],v_temp)
    axis_temp=axis_temp/np.linalg.norm(axis_temp)

    ###rotation about z axis to minimize x direction error
    R2=rot(np.array([0,0,np.sign(np.dot(axis_temp,v_norm))]),theta2)


    return np.dot(R1,R2)

def main():
    ###read interpolated curves in joint space
    col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
    data = read_csv("../data/from_cad/Curve_backproj_in_base_frame.csv", names=col_names)
    curve_x=data['X'].tolist()
    curve_y=data['Y'].tolist()
    curve_z=data['Z'].tolist()
    curve_direction_x=data['direction_x'].tolist()
    curve_direction_y=data['direction_y'].tolist()
    curve_direction_z=data['direction_z'].tolist()

    curve_backproj=np.vstack((curve_x, curve_y, curve_z)).T
    curve_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

    q_init=np.array([0.627463700138299,0.17976842821744082,0.5196590573281621,1.6053098733278601,-0.8935105128511388,0.9174696574156079])
    R_init=fwd(q_init).R

    pos_out=[]
    try:
        egm=rpi_abb_irc5.EGM()
        t1=time.time()
        idx=1
        arrived_init=False
        arrived_mid=False
        while True:
            res, state=egm.receive_from_robot(.01)
            
            if res:
                
                fb = state.robot_message.feedBack.cartesian

                if not arrived_init:
                    ###read in degrees
                    print('going to initial point',curve_backproj[0])
                    egm.send_to_robot_cart(curve_backproj[0],R2q(R_init))
                    if np.linalg.norm(np.array([fb.pos.x,fb.pos.y,fb.pos.z])-curve_backproj[0])<0.001:
                        arrived_init=True
                else:
                    pass
                    # joint_out.append(np.deg2rad(state.joint_angles))
                    # if np.linalg.norm(np.deg2rad(state.joint_angles)-curve_backproj_js[idx])>0.02:
                    #     ###send radians
                    #     egm.send_to_robot(curve_backproj_js[idx])
                    # else:
                    #     print('arrived',idx)
                    #     idx+=1
    except:
        traceback.print_exc()
        
        # joint_out=np.array(joint_out)
        # ###output to csv
        # df=DataFrame({'q0':joint_out[:,0],'q1':joint_out[:,1],'q2':joint_out[:,2],'q3':joint_out[:,3],'q4':joint_out[:,4],'q5':joint_out[:,5]})
        # df.to_csv('execution_egm.csv',header=False,index=False)

if __name__ == '__main__':
    main()
