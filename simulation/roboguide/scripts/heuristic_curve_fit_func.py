from math import radians,degrees
import numpy as np
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
    qall=robot.inv(pose)
    q = unwrapped_angle_check(ref_j,qall)
    wpr=R2wpr(pose.R)
    robt = joint2robtarget(q,robot,1,1,utool_num)
    for i in range(3):
        robt.trans[i]=pose.p[i]
        robt.rot[i]=wpr[i]
    return robt

def do_nothing(start_p,mid_p,end_p,tolerance,start_ref_j,mid_ref_j,end_ref_j,robot,utool_num):

    robt_all = []

    return

def lcl_fit(p1,p2,R1,R2,slope1,slope2):
    return 