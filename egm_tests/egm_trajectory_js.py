#!/usr/bin/env python

import rpi_abb_irc5
import time
import math
from pandas import *
import numpy as np

def main():
    ###read interpolated curves in joint space
    col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
    data = read_csv("../data/from_cad/Curve_backproj_js.csv", names=col_names)
    curve_q1=data['q1'].tolist()
    curve_q2=data['q2'].tolist()
    curve_q3=data['q3'].tolist()
    curve_q4=data['q4'].tolist()
    curve_q5=data['q5'].tolist()
    curve_q6=data['q6'].tolist()
    curve_backproj_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

    timestamp=[]
    joint_out=[]
    try:
        egm=rpi_abb_irc5.EGM()
        t1=time.time()
        idx=1
        arrived_init=False
        arrived_mid=False
        while True:
            res, state=egm.receive_from_robot(.01)
            
            if res:
                

                if not arrived_init:
                    ###read in degrees
                    print ("ID: " + str(state.robot_message.header.seqno) + " Joints: " + str(state.joint_angles))
                    q_cur=np.radians(state.joint_angles)
                    egm.send_to_robot(q_cur+(curve_backproj_js[0]-q_cur)/(10*np.linalg.norm(curve_backproj_js[0]-q_cur)))
                    if np.linalg.norm(np.deg2rad(state.joint_angles)-curve_backproj_js[0])<0.001:
                        arrived_init=True
                else:
                    timestamp.append(time.time())
                    joint_out.append(np.deg2rad(state.joint_angles))
                    if np.linalg.norm(np.deg2rad(state.joint_angles)-curve_backproj_js[idx])>0.01:
                        ###send radians
                        egm.send_to_robot(curve_backproj_js[idx])
                    else:
                        print('arrived',idx)
                        idx+=1
                    ###send radians
                    # egm.send_to_robot(curve_backproj_js[idx])
                    # idx+=1    
    except:
        timestamp=np.array(timestamp)
        joint_out=np.array(joint_out)
        ###output to csv
        df=DataFrame({'timestamp':timestamp,'q0':joint_out[:,0],'q1':joint_out[:,1],'q2':joint_out[:,2],'q3':joint_out[:,3],'q4':joint_out[:,4],'q5':joint_out[:,5]})
        df.to_csv('execution_egm.csv',header=False,index=False)

if __name__ == '__main__':
    main()
