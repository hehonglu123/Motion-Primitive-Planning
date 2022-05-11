from math import radians,degrees
import numpy as np
from pandas import read_csv,DataFrame
from math import sin,cos

from general_robotics_toolbox import *
import sys
from matplotlib import pyplot as plt

sys.path.append('../../../toolbox')
from robots_def import *
from lambda_calc import *
from error_check import *
sys.path.append('../../../circular_fit')
from toolbox_circular_fit import *
sys.path.append('../fanuc_toolbox')
from fanuc_client import *

def norm_vec(v):
    return v/np.linalg.norm(v)

def get_svJ(q_init,curve,curve_R,step,robot):
    curve = curve[::step]
    curve_R = curve_R[::step]
    all_svJ=[]
    
    q_last = q_init
    all_q = [q_last]
    for i in range(1,len(curve)):
        q_all=np.array(robot.inv(curve[i],curve_R[i]))
        if len(q_all)==0:
            all_svJ=[0]
            break
        q_last = unwrapped_angle_check(q_last,q_all)
        J=robot.jacobian(q_last)
        all_svJ.append(np.min(np.linalg.svd(J)[1]))
        all_q.append(q_last)
    
    return all_svJ,all_q

# generate curve with linear orientation
# robot=abb6640(d=50)
robot=m900ia(d=50)
# fanuc client
client = FANUCClient()
utool_num = 2

zone=100
speed=1000

start_p = np.array([2200,500, 1000])
end_p = np.array([2200, -500, 1000])
mid_p=(start_p+end_p)/2

data_dir='../data/data_collect/scene_1/'

all_max_error=[]
all_speed_var=[]
all_first_seg=[]
all_second_seg=[]
all_mismatch_ang=[]
all_first_seg_l=[]
all_second_seg_l=[]
all_first_pose=[]
all_second_pose=[]
all_end_pose=[]
all_tar_speed=[]

for ang in np.arange(0,90+0.001,0.5):
    print(ang)
    ##### create curve #####
    ###start rotation by 'ang' deg
    k=np.cross(end_p+np.array([0.1,0,0])-mid_p,start_p-mid_p)
    k=k/np.linalg.norm(k)
    theta=np.radians(ang)

    R=rot(k,theta)
    new_vec=R@(end_p-mid_p)
    new_end_p=mid_p+new_vec

    ###calculate lambda
    lam1_f=np.linalg.norm(mid_p-start_p)
    lam1=np.linspace(0,lam1_f,num=25001)
    lam_f=lam1_f+np.linalg.norm(mid_p-new_end_p)
    lam2=np.linspace(lam1_f,lam_f,num=25001)

    lam=np.hstack((lam1,lam2[1:]))

    #generate linear segment
    a1,b1,c1=lineFromPoints([lam1[0],start_p[0]],[lam1[-1],mid_p[0]])
    a2,b2,c2=lineFromPoints([lam1[0],start_p[1]],[lam1[-1],mid_p[1]])
    a3,b3,c3=lineFromPoints([lam1[0],start_p[2]],[lam1[-1],mid_p[2]])
    line1=np.vstack(((-a1*lam1-c1)/b1,(-a2*lam1-c2)/b2,(-a3*lam1-c3)/b3)).T

    a1,b1,c1=lineFromPoints([lam2[0],mid_p[0]],[lam2[-1],new_end_p[0]])
    a2,b2,c2=lineFromPoints([lam2[0],mid_p[1]],[lam2[-1],new_end_p[1]])
    a3,b3,c3=lineFromPoints([lam2[0],mid_p[2]],[lam2[-1],new_end_p[2]])
    line2=np.vstack(((-a1*lam2-c1)/b1,(-a2*lam2-c2)/b2,(-a3*lam2-c3)/b3)).T

    curve=np.vstack((line1,line2[1:]))

    R_init=Ry(np.radians(135))
    R_end=Ry(np.radians(90))
    # interpolate orientation
    R_all=[R_init]
    k,theta=R2rot(np.dot(R_end,R_init.T))
    # theta=np.pi/4 #force 45deg change
    for i in range(1,len(curve)):
        angle=theta*i/(len(curve)-1)
        R_temp=rot(k,angle)
        R_all.append(np.dot(R_temp,R_init))

    all_q_init = robot.inv(curve[0],R_init)
    argmin_q = -1
    max_min_svj = 0
    all_curve_js_sparse=[]
    for i in range(len(all_q_init)):
        all_svj,curve_js_sparse = get_svJ(all_q_init[i],curve,R_all,50,robot)
        min_svj= np.min(all_svj)
        if min_svj > max_min_svj:
            max_min_svj=min_svj
            argmin_q=i
        all_curve_js_sparse.append(curve_js_sparse)
    if argmin_q==-1:
        print("No solution")
        continue
    curve_js_sparse=all_curve_js_sparse[argmin_q]

    # tp program
    # move to start
    tp_pre = TPMotionProgram()
    j0 = jointtarget(1,1,utool_num,np.degrees(curve_js_sparse[0]),[0]*6)
    tp_pre.moveJ(j0,50,'%',-1)
    # robt_start = joint2robtarget(curve_js[0],robot,1,1,utool_num)
    robt_start = jointtarget(1,1,utool_num,np.degrees(curve_js_sparse[0]),[0]*6)
    tp_pre.moveL(robt_start,5,'mmsec',-1)
    client.execute_motion_program(tp_pre)
    # print(robt_start)
    # print(curve_js[0])

    tp = TPMotionProgram()
    joint_mid = unwrapped_angle_check(curve_js_sparse[int((len(curve_js_sparse)+1)/2)],robot.inv(curve[int((len(curve)+1)/2)],R_all[int((len(R_all)+1)/2)]))
    # robt_mid = joint2robtarget(joint_mid,robot,1,1,utool_num)
    robt_mid = jointtarget(1,1,utool_num,np.degrees(joint_mid),[0]*6)
    tp.moveL(robt_mid,speed,'mmsec',zone)
    joint_end = unwrapped_angle_check(curve_js_sparse[-1],robot.inv(curve[-1],R_all[-1]))
    # robt_end = joint2robtarget(joint_end,robot,1,1,utool_num)
    robt_end = jointtarget(1,1,utool_num,np.degrees(joint_end),[0]*6)
    tp.moveL(robt_end,speed,'mmsec',-1)

    # execute 
    res = client.execute_motion_program(tp)
    res_string = res.decode('utf-8')
    timestamp=[]
    curve_exe_js=[]
    for res_lines in res_string.split("\n")[1:]:
        if res_lines=='':
            break
        this_q_stamp = []
        for j_str in res_lines.split(", "):
            this_q_stamp.append(float(j_str))
        curve_exe_js.append(np.array(this_q_stamp[1:]))
        timestamp.append(float(this_q_stamp[0]))
    curve_exe_js=np.radians(curve_exe_js)
    timestamp=np.array(timestamp)/1000.
    
    # find error and speed norm
    act_speed=[]
    lam_exec=[0]
    curve_exe=[]
    curve_exe_R=[]
    dont_show_id=[]
    last_cont = False
    for i in range(len(curve_exe_js)):
        this_q = curve_exe_js[i]
        if i>5 and i<len(curve_exe_js)-5:
            # if the recording is not fast enough
            # then having to same logged joint angle
            # do interpolation for estimation
            if np.all(this_q==curve_exe_js[i+1]):
                dont_show_id=np.append(dont_show_id,i).astype(int)
                last_cont = True
                continue

        robot_pose=robot.fwd(this_q)
        curve_exe.append(robot_pose.p)
        curve_exe_R.append(robot_pose.R)
        if i>0:
            lam_exec.append(lam_exec[-1]+np.linalg.norm(curve_exe[-1]-curve_exe[-2]))
        try:
            if timestamp[-1]!=timestamp[-2]:
                if last_cont:
                    timestep=timestamp[i]-timestamp[i-2]
                else:
                    timestep=timestamp[i]-timestamp[i-1]
                act_speed.append(np.linalg.norm(curve_exe[-1]-curve_exe[-2])/timestep)
        except IndexError:
            pass
        
        last_cont = False
    curve_exe=np.array(curve_exe)
    curve_exe_R=np.array(curve_exe_R)
    
    error_all=calc_all_error(curve_exe,curve)

    max_error = np.max(error_all)

    speed_first_half = np.argmax(act_speed[:int(len(act_speed)/2-3)])
    speed_second_half = np.argmax(act_speed[int(len(act_speed)/2+3):])+int(len(act_speed)/2+3)
    speed_var = np.min([0,speed-np.min(act_speed[speed_first_half:speed_second_half])])

    all_max_error.append(max_error)
    all_speed_var.append(speed_var)
    all_first_seg.append(1)
    all_second_seg.append(1)
    all_mismatch_ang.append(ang)
    all_first_seg_l.append(lam1_f)
    all_second_seg_l.append(lam_f-lam1_f)
    all_first_pose.append(curve_js_sparse[0])
    all_second_pose.append(joint_mid)
    all_end_pose.append(joint_end)
    all_tar_speed.append(1000)
    with open(data_dir+'all_max_error.npy','wb') as f:
        np.save(f,all_max_error)
    with open(data_dir+'all_speed_var.npy','wb') as f:
        np.save(f,all_speed_var)
    with open(data_dir+'all_first_seg.npy','wb') as f:
        np.save(f,all_first_seg)
    with open(data_dir+'all_second_seg.npy','wb') as f:
        np.save(f,all_second_seg)
    with open(data_dir+'all_mismatch_ang.npy','wb') as f:
        np.save(f,all_mismatch_ang)
    with open(data_dir+'all_first_seg_l.npy','wb') as f:
        np.save(f,all_first_seg_l)
    with open(data_dir+'all_second_seg_l.npy','wb') as f:
        np.save(f,all_second_seg_l)
    with open(data_dir+'all_first_pose.npy','wb') as f:
        np.save(f,all_first_pose)
    with open(data_dir+'all_second_pose.npy','wb') as f:
        np.save(f,all_second_pose)
    with open(data_dir+'all_end_pose.npy','wb') as f:
        np.save(f,all_end_pose)
    with open(data_dir+'all_tar_speed.npy','wb') as f:
        np.save(f,all_tar_speed)