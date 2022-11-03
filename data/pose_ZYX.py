
import numpy as np
from pandas import *
import sys
from general_robotics_toolbox import *
from utils import *
from robots_def import *


dataset='wood/'
solution_dir='dual_arm/diffevo_pose3/'
data_dir=dataset+solution_dir

###reference frame transformation
curve_pose=np.loadtxt(data_dir+'base.csv',delimiter=',')

print(curve_pose[:-1,-1])
print(np.degrees(rotationMatrixToEulerAngles(curve_pose[:3,:3])))

