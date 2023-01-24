import numpy as np
from pandas import *
import sys, traceback, time
from general_robotics_toolbox import *
from fanuc_motion_program_exec_client import *
import matplotlib.pyplot as plt
import os
import yaml

sys.path.append('../../../toolbox/')
from utils import *
from robots_def import *

robot1=m10ia(d=50)
robot2=lrmate200id()
# print(robot1.P[:,0])
r1l1_r2l1_R = wpr2R([0.25,0.175,89.85]) # [w,p,r]
r1l1_r2l1 = Transform(r1l1_r2l1_R,[1237.9,-712.818,306.345])
r1b_r1l1 = Transform(np.eye(3),robot1.P[:,0])
r2b_r2l1 = Transform(np.eye(3),robot2.P[:,0])

r1b_r2b = (r1b_r1l1*r1l1_r2l1)*(r2b_r2l1.inv())
print(r1b_r2b)
print(R2wpr(r1b_r2b.R))