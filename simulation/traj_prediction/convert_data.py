from curses import qiflush
from urllib.parse import quote_plus
import numpy as np
from numpy.linalg import norm
import general_robotics_toolbox as rox
from general_robotics_toolbox import general_robotics_toolbox_invkin as roxinv
from math import ceil,cos, sin
from pandas import *
import time
import os
import csv

import sys
sys.path.append('../../toolbox')
from robots_def import *

from nominal_traj import AnalyticalPredictor

dt = 0.004
robot = abb6640(R_tool=Ry(np.radians(90)),p_tool=np.array([0,0,0]))

src_folder = '/media/eric/Transcend/Motion-Primitive-Planning/simulation/traj_prediction/data/data_L_z10/'
des_folder = '/media/eric/Transcend/Motion-Primitive-Planning/simulation/traj_prediction/data/data_L_z10_split/'

# data_cnt = 0
# data_ptr = -1
data_cnt = 152
data_ptr = 326
max_data = 3000

Tf = 25
ap = AnalyticalPredictor(1,Tf)

while data_cnt < max_data:
    data_ptr += 1

    # load motion instructions
    col_names=['motion','p1_x','p1_y','p1_z','p1_qw','p1_qx','p1_qy','p1_qz', \
                'p1_cf1','p1_cf2','p1_cf3','p1_cf4',\
                'p2_x','p2_y','p2_z','p2_qw','p2_qx','p2_qy','p2_qz', \
                'p2_cf1','p2_cf2','p2_cf3','p2_cf4',\
                'p3_x','p3_y','p3_z','p3_qw','p3_qx','p3_qy','p3_qz', \
                'p3_cf1','p3_cf2','p3_cf3','p3_cf4',\
                'vel','zone'] 
    motions = read_csv(src_folder+"train_data/"+str(data_ptr)+".txt", names=col_names)
    p1 = np.array([motions['p1_x'].tolist(),motions['p1_y'].tolist(),motions['p1_z'].tolist()])
    R1 = rox.q2R(np.array([motions['p1_qw'].tolist(),motions['p1_qx'].tolist(),motions['p1_qy'].tolist(),motions['p1_qz'].tolist()]))
    T1 = rox.Transform(R1,p1)
    p2 = np.array([motions['p2_x'].tolist(),motions['p2_y'].tolist(),motions['p2_z'].tolist()])
    R2 = rox.q2R(np.array([motions['p2_qw'].tolist(),motions['p2_qx'].tolist(),motions['p2_qy'].tolist(),motions['p2_qz'].tolist()]))
    T2 = rox.Transform(R2,p2)
    p3 = np.array([motions['p3_x'].tolist(),motions['p3_y'].tolist(),motions['p3_z'].tolist()])
    R3 = rox.q2R(np.array([motions['p3_qw'].tolist(),motions['p3_qx'].tolist(),motions['p3_qy'].tolist(),motions['p3_qz'].tolist()]))
    T3 = rox.Transform(R3,p3)

    vel_profile = motions['vel'].tolist()[0]
    zone_profile = motions['zone'].tolist()[0]

    if zone_profile == 10:
        z_tcp = 10
        z_ori = 15
    elif zone_profile == 1:
        z_tcp = 1
        z_ori = 1

    # load logged joints
    col_names=['timestamp','cmd_num','q1', 'q2', 'q3','q4', 'q5', 'q6'] 
    data = read_csv(src_folder+"train_data_gt/"+str(data_ptr)+".csv", names=col_names,skiprows = 1)
    curve_q1=data['q1'].tolist()
    curve_q2=data['q2'].tolist()
    curve_q3=data['q3'].tolist()
    curve_q4=data['q4'].tolist()
    curve_q5=data['q5'].tolist()
    curve_q6=data['q6'].tolist()
    joint_angles=np.deg2rad(np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T)
    # skip the data that directly has cmd == 3
    if data['cmd_num'].tolist()[0] > 2:
        print("Not this one")
        continue
    curve_x=np.array([])
    curve_y=np.array([])
    curve_z=np.array([])
    for i in range(len(joint_angles)):
        T = robot.fwd(joint_angles[i])
        curve_x = np.append(curve_x,T.p[0])
        curve_y = np.append(curve_y,T.p[1])
        curve_z = np.append(curve_z,T.p[2])

    try: 
        save_folder_input = des_folder+str(data_ptr)
        os.mkdir(save_folder_input)
        save_folder_input = des_folder+str(data_ptr)+'/input'
        os.mkdir(save_folder_input)
        save_folder_label = des_folder+str(data_ptr)+'/label'
        os.mkdir(save_folder_label)
    except OSError as error: 
        print(error) 
    second_half = False
    for q_start_i in range(len(joint_angles)):
        
        if second_half:
            q_input,_ = ap.predict(joint_angles[q_start_i],T2,T3,T3,vel_profile,0,0) # second half has fine zone
        else:
            q_input,_ = ap.predict(joint_angles[q_start_i],T1,T2,T3,vel_profile,z_tcp,z_ori)
        

        for q_append_i in range(q_start_i,q_start_i-Tf-1,-1):
            if q_append_i < 0:
                q_input.insert(0,joint_angles[0])
            else:
                q_input.insert(0,joint_angles[q_append_i])

        if q_start_i < (len(joint_angles)-Tf):
            q_output = list(joint_angles[q_start_i+1:q_start_i+1+Tf])
        else:
            q_output = list(joint_angles[q_start_i+1:])
        while len(q_output) < Tf:
            q_output.append(joint_angles[-1])

        q_input = np.rad2deg(q_input)
        q_output = np.rad2deg(q_output)
        save_input_file_name = save_folder_input+'/'+str(q_start_i)+'.csv'
        with open(save_input_file_name,"w",newline='') as f:
            w = csv.writer(f)
            w.writerows(q_input)
        save_label_file_name = save_folder_label+'/'+str(q_start_i)+'.csv'
        with open(save_label_file_name,"w",newline='') as f:
            w = csv.writer(f)
            w.writerows(q_output)
        
        if not second_half:
            this_p = np.array([curve_x[q_start_i],curve_y[q_start_i],curve_z[q_start_i]])
            next_p = np.array([curve_x[q_start_i+1],curve_y[q_start_i+1],curve_z[q_start_i+1]])
            dthis_t2 = norm(this_p-T2.p)
            dnext_t2 = norm(next_p-T2.p)

            # if this p in zone but next dont
            # else if within distance, next p is farther than this p
            if (dthis_t2<=z_tcp) :
                if (dnext_t2>z_tcp):
                    second_half = True
            elif (dthis_t2<=10) and ((dthis_t2<=dnext_t2)):
                second_half = True
            else:
                pass
    
    data_cnt += 1
    print("Data Finished:",data_cnt)