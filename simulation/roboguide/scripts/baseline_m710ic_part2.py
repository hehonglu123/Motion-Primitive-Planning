from math import radians,pi
import numpy as np
from pandas import *
import yaml
from matplotlib import pyplot as plt

from general_robotics_toolbox import *
from general_robotics_toolbox.general_robotics_toolbox_invkin import *
import sys


# from simulation.roboguide.fanuc_toolbox.fanuc_client import FANUCClient, TPMotionProgram, joint2robtarget, jointtarget, robtarget
# from toolbox.robots_def import arb_robot, m900ia
sys.path.append('../../../toolbox')
from robots_def import *
from utils import *
from lambda_calc import *
sys.path.append('../../../greedy_fitting')
from greedy import *


# obj_type='wood'
obj_type='blade'
data_dir='../data/baseline_m710ic/'+obj_type+'/'

robot=m710ic(d=50)
# curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
###read actual curve
curve_js = read_csv(data_dir+"Curve_js.csv",header=None).values
curve_js=np.array(curve_js)

thresholds=[0.1,0.2,0.5,0.9]

for threshold in thresholds:
    greedy_fit_obj=greedy_fit(robot,curve_js,threshold)
    greedy_fit_obj.primitives={'movel_fit':greedy_fit_obj.movel_fit_greedy}
    breakpoints,primitives_choices,points=greedy_fit_obj.fit_under_error()