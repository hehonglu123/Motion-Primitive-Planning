from math import radians,degrees
from tokenize import Pointfloat
import numpy as np
from numpy.linalg import norm
from pandas import read_csv,DataFrame
from math import sin,cos
from copy import deepcopy

from general_robotics_toolbox import *
import sys
from matplotlib import pyplot as plt
sys.path.append('../fanuc_toolbox')
from fanuc_client import *
sys.path.append('../../../toolbox')
from robots_def import *

def norm_vec(v):
    return v/np.linalg.norm(v)

def get_robtarget(pose,ref_j,robot,utool_num):
    qall=robot.inv(pose.p,pose.R)
    q = unwrapped_angle_check(ref_j,qall)
    wpr=R2wpr(pose.R)
    robt = joint2robtarget(q,robot,1,1,utool_num)
    for i in range(3):
        robt.trans[i]=pose.p[i]
        robt.rot[i]=wpr[i]
    return robt

def push_tolerance(tolerance,start_p,mid_p,end_p):
    st2en=norm_vec(end_p.p-start_p.p)
    st2md=norm_vec(mid_p.p-start_p.p)
    md2end=norm_vec(end_p.p-mid_p.p)
    plane_norm = norm_vec(np.cross(st2en,st2md))
    start_push_dir = np.cross(plane_norm,st2md)
    mid_push_dir = np.cross(plane_norm,st2en)
    end_push_dir = np.cross(plane_norm,md2end)

    ang = np.arccos(np.dot(st2md,md2end))

    start_p.p = start_p.p+start_push_dir*tolerance
    start_mid_p = deepcopy(start_p)
    start_mid_p.p = mid_p.p+start_push_dir*tolerance
    end_p.p = end_p.p+end_push_dir*tolerance
    end_mid_p = deepcopy(end_p)
    end_mid_p.p = mid_p.p+start_push_dir*tolerance
    mid_p.p = mid_p.p+mid_push_dir*tolerance*(1/cos(ang/2))

    st2md_k,st2md_th = R2rot(np.matmul(start_p.R.T,mid_p.R))
    ed2md_k,ed2md_th = R2rot(np.matmul(end_p.R.T,mid_p.R))

    start_mid_p.R = np.matmul(start_p.R,rot(st2md_k,st2md_th*(norm(start_mid_p.p-start_p.p)/norm(mid_p.p-start_p.p))))
    end_mid_p.R = np.matmul(end_p.R,rot(ed2md_k,ed2md_th*(norm(end_mid_p.p-end_p.p)/norm(mid_p.p-end_p.p))))

    return start_p,mid_p,end_p,start_mid_p,end_mid_p

def find_arc(start_p,mid_p,end_p,align_p):

    st2en=norm_vec(end_p.p-start_p.p)
    st2md=norm_vec(mid_p.p-start_p.p)
    md2en=norm_vec(end_p.p-mid_p.p)
    plane_norm = norm_vec(np.cross(st2en,st2md))
    tar_per_norm = np.cross(plane_norm,st2en)

    st_k = np.dot((align_p-start_p.p),tar_per_norm)/np.dot(st2md,tar_per_norm)
    st_k_rat = st_k/np.linalg.norm(mid_p.p-start_p.p)
    en_k = np.dot((align_p-end_p.p),tar_per_norm)/np.dot(-md2en,tar_per_norm)
    en_k_rat = st_k/np.linalg.norm(end_p.p-mid_p.p)

    st2md_k,st2md_th = R2rot(np.matmul(start_p.R.T,mid_p.R))
    ed2md_k,ed2md_th = R2rot(np.matmul(end_p.R.T,mid_p.R))

    arc_start_p = Transform(np.matmul(start_p.R,rot(st2md_k,st2md_th*st_k_rat)),start_p.p+st2md*st_k)
    arc_end_p = Transform(np.matmul(end_p.R,rot(ed2md_k,ed2md_th*en_k_rat)),end_p.p+(-md2en)*en_k)
    arc_c = np.matmul(np.linalg.pinv([st2md,md2en,plane_norm]),\
        [np.dot(arc_start_p.p,st2md),np.dot(arc_end_p.p,md2en),np.dot(arc_start_p.p,plane_norm)])
    arc_r = np.linalg.norm(arc_start_p.p-arc_c)
    arc_ang = np.arccos(np.dot(norm_vec(arc_start_p.p-arc_c), norm_vec(arc_end_p.p-arc_c)))

    return arc_start_p,arc_end_p,arc_c,arc_r,arc_ang,-plane_norm

def addmid_h(h,start_p,mid_p,end_p,start_ref_j,mid_ref_j,end_ref_j,robot,utool_num):

    st2en=norm_vec(end_p.p-start_p.p)
    st2md=norm_vec(mid_p.p-start_p.p)
    md2end=norm_vec(end_p.p-mid_p.p)
    plane_norm = norm_vec(np.cross(st2en,st2md))
    mid_push_dir = np.cross(plane_norm,st2en)

    start_mid_p = deepcopy(start_p)
    end_mid_p = deepcopy(end_p)
    start_mid_p.p = (mid_p.p+start_p.p)/2
    end_mid_p.p = (mid_p.p+end_mid_p.p)/2

    # push mid_p by h
    mid_p.p = mid_p.p+mid_push_dir*h

    stmd_ratio = norm(start_mid_p.p-start_p.p)/(norm(mid_p.p-start_mid_p.p)+norm(start_mid_p.p-start_p.p))
    enmd_ratio = norm(end_mid_p.p-end_p.p)/(norm(mid_p.p-end_mid_p.p)+norm(end_mid_p.p-end_p.p))
    # add start_mid and end_mid R according to length ratio
    st2md_k,st2md_th = R2rot(np.matmul(start_p.R.T,mid_p.R))
    ed2md_k,ed2md_th = R2rot(np.matmul(end_p.R.T,mid_p.R))
    start_mid_p.R = np.matmul(start_p.R,rot(st2md_k,st2md_th*stmd_ratio))
    end_mid_p.R = np.matmul(end_p.R,rot(ed2md_k,ed2md_th*enmd_ratio))

    # starting
    robt_all = [get_robtarget(start_p,start_ref_j,robot,utool_num)]
    motion_all=['L']
    robt_all.append(get_robtarget(start_mid_p,start_ref_j,robot,utool_num))
    motion_all.append('L')
    robt_all.append(get_robtarget(mid_p,mid_ref_j,robot,utool_num))
    motion_all.append('L')
    robt_all.append(get_robtarget(end_mid_p,end_ref_j,robot,utool_num))
    motion_all.append('L')
    robt_all.append(get_robtarget(end_p,end_ref_j,robot,utool_num))
    return robt_all,motion_all

def addC(tolerance,start_p,mid_p,end_p,start_ref_j,mid_ref_j,end_ref_j,robot,utool_num):
    old_midp = deepcopy(mid_p)
    start_p,mid_p,end_p,start_mid_p,end_mid_p = push_tolerance(tolerance,start_p,mid_p,end_p)

    arc_start_p,arc_end_p,arc_c,arc_r,arc_ang,rot_ax=find_arc(start_p,mid_p,end_p,old_midp.p)
    # starting
    robt_all = [get_robtarget(start_p,start_ref_j,robot,utool_num)]
    motion_all=['L']
    robt_all.append(get_robtarget(arc_start_p,mid_ref_j,robot,utool_num))
    # C
    motion_all.append('C')
    robt_c = []
    k,theta = R2rot(np.matmul(arc_start_p.R.T,arc_end_p.R))
    for i in range(2):
        position = np.matmul(rot(rot_ax,float(i+1)/2*arc_ang),(arc_start_p.p-arc_c))+arc_c
        orientation = np.matmul(arc_start_p.R,rot(k,float(i+1)/2*theta))
        robt_c.append(get_robtarget(Transform(orientation,position),mid_ref_j,robot,utool_num))
    robt_all.append(robt_c)
    # ending
    motion_all.append('L')
    robt_all.append(get_robtarget(end_p,end_ref_j,robot,utool_num))
    return robt_all,motion_all

def addL(l_num,tolerance,start_p,mid_p,end_p,start_ref_j,mid_ref_j,end_ref_j,robot,utool_num):
    old_midp = deepcopy(mid_p)
    start_p,mid_p,end_p,start_mid_p,end_mid_p = push_tolerance(tolerance,start_p,mid_p,end_p)

    arc_start_p,arc_end_p,arc_c,arc_r,arc_ang,rot_ax=find_arc(start_p,mid_p,end_p,old_midp.p)
    # starting
    robt_all = [get_robtarget(start_p,start_ref_j,robot,utool_num)]
    motion_all=['L']
    # arc region
    robt_all.append(get_robtarget(arc_start_p,mid_ref_j,robot,utool_num))
    motion_all=['L']
    k,theta = R2rot(np.matmul(arc_start_p.R.T,arc_end_p.R))
    for i in range(l_num):
        position = np.matmul(rot(rot_ax,(i+1)/l_num*arc_ang),(arc_start_p.p-arc_c))+arc_c
        orientation = np.matmul(arc_start_p.R,rot(k,(i+1)/l_num*theta))
        robt_all.append(get_robtarget(Transform(orientation,position),mid_ref_j,robot,utool_num))
        motion_all.append('L')
    # ending
    robt_all.append(get_robtarget(end_p,end_ref_j,robot,utool_num))
    return robt_all,motion_all

def exploit_tolerance(tolerance,start_p,mid_p,end_p,start_ref_j,mid_ref_j,end_ref_j,robot,utool_num):

    start_p,mid_p,end_p,start_mid_p,end_mid_p = push_tolerance(tolerance,start_p,mid_p,end_p)
    return do_nothing(start_p,mid_p,end_p,start_ref_j,mid_ref_j,end_ref_j,robot,utool_num)

def do_nothing(start_p,mid_p,end_p,start_ref_j,mid_ref_j,end_ref_j,robot,utool_num):

    robt_all = []
    motion_all=[]
    robt_all.append(get_robtarget(start_p,start_ref_j,robot,utool_num))
    motion_all.append('L')
    robt_all.append(get_robtarget(mid_p,mid_ref_j,robot,utool_num))
    motion_all.append('L')
    robt_all.append(get_robtarget(end_p,end_ref_j,robot,utool_num))

    return robt_all,motion_all

# start_p = np.array([2200,500, 1000])
# end_p = np.array([2200, -500, 1000])
# mid_p=(start_p+end_p)/2
# ang=30
# ##### create curve #####
# ###start rotation by 'ang' deg
# k=np.cross(end_p+np.array([0.1,0,0])-mid_p,start_p-mid_p)
# k=k/np.linalg.norm(k)
# theta=np.radians(ang)

# R=rot(k,theta)
# new_vec=R@(end_p-mid_p)
# new_end_p=mid_p+new_vec

# old_midp=mid_p
# start_p,mid_p,end_p=push_tolerance(1,Transform(np.eye(3),start_p),Transform(np.eye(3),mid_p),Transform(np.eye(3),new_end_p))
# arc_start_p,arc_end_p,arc_c,arc_r,arc_ang,rot_ax=find_arc(start_p,mid_p,end_p,old_midp)
# print(arc_start_p,arc_end_p,arc_c,arc_r,arc_ang,rot_ax)
# position = np.matmul(rot(rot_ax,arc_ang),(arc_start_p.p-arc_c))+arc_c
# print(position)
# print(arc_end_p.p)
