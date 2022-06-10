from distutils.command.config import config
from matplotlib.pyplot import contour
from matplotlib.transforms import Transform
import numpy as np
from pandas import *
import general_robotics_toolbox as rox
from general_robotics_toolbox import general_robotics_toolbox_invkin as roxinv
from random import randint, sample, uniform
from math import cos,sin,pi,radians
import random
import csv
import time

import sys

sys.path.append('../../toolbox')
from robots_def import *

from abb_motion_program_exec_client import *

def quadrant(q):
    temp=np.ceil(np.array([q[0],q[3],q[5]])/(np.pi/2))-1
    
    if q[4] < 0:
        last = 1
    else:
        last = 0

    return np.hstack((temp,[last])).astype(int)

# parameters
# total_data_num = 30000
total_data_num = 30000
point_range = [1000,4000]
trans_x_diff = [-100,100]
trans_y_diff = [-100,100]
trans_z_diff = [-100,100]
# rot_x_diff = [-45,45]
# rot_y_diff = [-45,45]
# rot_z_diff = [-45,45]
rot_ang_diff = [-10,10]

# load curve
col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
data = read_csv("../../data/from_ge/Curve_backproj_js.csv", names=col_names)
curve_q1=data['q1'].tolist()
curve_q2=data['q2'].tolist()
curve_q3=data['q3'].tolist()
curve_q4=data['q4'].tolist()
curve_q5=data['q5'].tolist()
curve_q6=data['q6'].tolist()
curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T
robot=abb6640(R_tool=Ry(np.radians(90)),p_tool=np.array([0,0,0]))
    
curve_center_id = 32684
cur_center_T = robot.fwd(curve_js[curve_center_id])
curve_trans = rox.Transform(np.eye(3),cur_center_T.p)

client = MotionProgramExecClient()

# data acquisition
data_cnt = 0



total_len = len(curve_js)
print(total_len)
while data_cnt < total_data_num:
    start_point_id = randint(0,total_len)
    direction = sample([-1,1],k=1)[0]

    # find two goal points
    pt_ids = [start_point_id]
    cont_flag = False
    for pt in range(2):
        up_limit = np.max([(0-pt_ids[pt])/direction,(total_len-1-pt_ids[pt])/direction])
        upper_bound = np.min([up_limit,point_range[1]])
        if up_limit < point_range[0]:
            cont_flag=True
            break
        pt_ids.append(pt_ids[pt] + direction*randint(point_range[0],upper_bound))
    if cont_flag:
        continue

    print(pt_ids)
    
    # do a transform to increase richness of the dataset.
    rand_k_th,rand_k_phi = random.uniform(-pi,pi),random.uniform(-pi,pi)
    rand_k = np.array([sin(rand_k_th)*cos(rand_k_phi),sin(rand_k_th)*sin(rand_k_phi),cos(rand_k_th)])
    rand_th = random.uniform(radians(rot_ang_diff[0]),radians(rot_ang_diff[1]))
    rand_R = rox.rot(rand_k,rand_th)
    rand_p = np.array([random.uniform(trans_x_diff[0],trans_x_diff[1]),\
            random.uniform(trans_y_diff[0],trans_y_diff[1]),\
            random.uniform(trans_z_diff[0],trans_z_diff[1])])
    rand_T = rox.Transform(rand_R,rand_p)
    # rand_T = rox.Transform(np.eye(3),[0,0,0])

    # convert from joint space to cartesian space
    T_path = []
    for ptid in pt_ids:
        T_path.append(cur_center_T*rand_T*cur_center_T.inv()*robot.fwd(curve_js[ptid]))

    vel_profile = sample([50,500],k=1)[0]
    # zone_profile = sample([0,1,10],k=1)[0]
    zone_profile = 10

    # pick a zone profile
    # Rule
    # 1. Choose the one match the previous zone
    # 2. Choose the one with the most "0" quadrant
    q_start_all = robot.inv(T_path[0].p,T_path[0].R)
    if len(q_start_all) == 0:
        continue
    quad_start_all = []
    quad_start = None
    q_start = None
    quad_start_origin = quadrant(curve_js[pt_ids[0]])
    for q in q_start_all:
        quad_start_all.append(quadrant(q))
        if np.all(quad_start_all[-1] == quad_start_origin):
            quad_start = quad_start_all[-1]
            q_start = q
            break
    if quad_start is None:
        qid = np.argmin(np.linalg.norm(quad_start_all,1,axis=1))
        quad_start = quad_start_all[qid]
        q_start = q_start_all[qid]
    config_profile_all = []
    config_profile_all.append(quad_start)
    # get the quadrant of all q based on the first one
    step_num = 10 # divided in to step_num
    not_this_point_flag = False
    for T_id in range(1,len(T_path)):
        p_diff = T_path[T_id].p-T_path[T_id-1].p
        th_k,th_diff = rox.R2rot(np.matmul(T_path[T_id-1].R.T,T_path[T_id].R))

        q_prop = q_start
        for step_i in range(1,step_num+1):
            ang_prop = step_i*th_diff/step_num
            p_prop = T_path[T_id-1].p + step_i*p_diff/step_num
            T_prop = rox.Transform(np.dot(T_path[T_id-1].R,rox.rot(th_k,ang_prop)),p_prop) 
            q_prop_all = roxinv.robot6_sphericalwrist_invkin(robot.robot_def,T_prop,q_prop)
            if len(q_prop_all) == 0:
                not_this_point_flag = True
                break
            q_prop = q_prop_all[0]
        if not_this_point_flag:
            break

        config_profile_all.append(quadrant(q_prop))
    if not_this_point_flag:
        continue

    # log joint trajectory from robotstudio

    mp = MotionProgram()
    
    vel = None
    if vel_profile == 50:
        vel = v50
    elif vel_profile == 500:
        vel = v500
    
    zone = None
    if zone_profile == 1:
        zone = z1
    elif zone_profile == 10:
        zone = z10
    else:
        zone = fine
    

    # 1. Move to the first pose using MoveAsJ
    q_start_deg = np.rad2deg(q_start)
    jointt = jointtarget([q_start_deg[0],q_start_deg[1],q_start_deg[2],q_start_deg[3],q_start_deg[4],q_start_deg[5]],[0]*6)
    mp.MoveAbsJ(jointt,v5000,fine)
    # 2. Move to the second and third pose using MoveL
    for T_id in range(1,len(T_path)-1):
        point = T_path[T_id].p
        quat = rox.R2q(T_path[T_id].R)
        cf = config_profile_all[T_id]
        robt = robtarget([point[0], point[1], point[2]], [ quat[0], quat[1], quat[2], quat[3]], confdata(cf[0],cf[1],cf[2],cf[3]),[9E+09]*6)
        mp.MoveL(robt,vel,zone)
    point = T_path[-1].p
    quat = rox.R2q(T_path[-1].R)
    cf = config_profile_all[-1]
    robt = robtarget([point[0], point[1], point[2]], [ quat[0], quat[1], quat[2], quat[3]], confdata(cf[0],cf[1],cf[2],cf[3]),[9E+09]*6)
    mp.MoveL(robt,vel,fine) # last point should be fine

    # print(mp.get_program_rapid())

    # save ground truth joint trajectory
    try:
        log_results = client.execute_motion_program(mp)
    except AssertionError as e:
        print("ABB Error",e)
        time.sleep(3)
        continue
    # log_results = client.execute_motion_program(mp)
    log_results_str = log_results.decode('ascii')

    log_results_dict = {}
    rows = log_results_str.split("\r\n")
    for row in rows[:-1]:
        if len(log_results_dict) == 0:
            log_results_dict['timestamp']=[]
            log_results_dict['cmd_num']=[]
            log_results_dict['J1']=[]
            log_results_dict['J2']=[]
            log_results_dict['J3']=[]
            log_results_dict['J4']=[]
            log_results_dict['J5']=[]
            log_results_dict['J6']=[]
            continue
        col = row.split(", ")
        if float(col[1]) < 2:
            continue
        log_results_dict['timestamp'].append(float(col[0]))
        log_results_dict['cmd_num'].append(float(col[1]))
        log_results_dict['J1'].append(float(col[2]))
        log_results_dict['J2'].append(float(col[3]))
        log_results_dict['J3'].append(float(col[4]))
        log_results_dict['J4'].append(float(col[5]))
        log_results_dict['J5'].append(float(col[6]))
        log_results_dict['J6'].append(float(col[7]))

    # print(np.where(np.array(log_results_dict['cmd_num'])==2)[0])
    # print(np.where(np.array(log_results_dict['cmd_num'])==2)[0][0])
    # start_id = np.where(np.array(log_results_dict['cmd_num'])==2)[0][0]
    # log_results_dict['timestamp'] = log_results_dict['timestamp'][start_id:]
    # log_results_dict['cmd_num'] = log_results_dict['cmd_num'][start_id:]
    # log_results_dict['joint_angle'] = log_results_dict['joint_angle'][start_id:]

    save_file_name = 'data_L_z10/train_data_gt/'+str(data_cnt)+'.csv'
    with open(save_file_name,"w",newline='') as f:
        w = csv.writer(f)
        w.writerow(log_results_dict.keys())
        w.writerows(zip(*log_results_dict.values()))
    
    input_str = ""
    # add motion primitive type
    input_str += 'L,'
    # add robot taget 1~3
    for Tid in range(len(T_path)):
        for p in T_path[Tid].p:
            input_str += str(p)+','
        quat=rox.R2q(T_path[Tid].R)
        for qua in quat:
            input_str += str(qua)+','
        cfg = config_profile_all[Tid]
        for cf in cfg:
            input_str += str(cf)+','
    # add vel
    input_str += str(vel_profile)+','
    # add zone
    input_str += str(zone_profile)

    save_file_name = 'data_L_z10/train_data/'+str(data_cnt)+'.txt'
    with open(save_file_name,"w") as f:
        f.write(input_str)

    print(data_cnt)
    data_cnt += 1