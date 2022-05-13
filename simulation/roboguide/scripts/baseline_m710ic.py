from math import radians
import numpy as np
from pandas import read_csv

from general_robotics_toolbox import *
from general_robotics_toolbox.general_robotics_toolbox_invkin import *
import sys

# from simulation.roboguide.fanuc_toolbox.fanuc_client import FANUCClient, TPMotionProgram, joint2robtarget, jointtarget, robtarget
# from toolbox.robots_def import arb_robot, m900ia
sys.path.append('../../../toolbox')
from robots_def import *
sys.path.append('../fanuc_toolbox')
from fanuc_client import *
sys.path.append('../../../constraint_solver')
from qp_resolve import *
sys.path.append('../../../greedy_fitting')
from greedy_poly import *
sys.path.append('../../../data')
from baseline import *

# define robot
robot=m710ic(d=50)

# the original curve in Cartesian space
col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
data = read_csv("../../../data/wood/Curve_dense.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve_direction_x=data['direction_x'].tolist()
curve_direction_y=data['direction_y'].tolist()
curve_direction_z=data['direction_z'].tolist()
curve=np.vstack((curve_x, curve_y, curve_z)).T
curve_normal=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

# put the curve in the best pose relative to the robot
print(pose_opt(robot,curve,curve_normal))