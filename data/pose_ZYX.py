
import numpy as np
from pandas import *
import sys
from general_robotics_toolbox import *
from utils import *
from robots_def import *


dataset='from_NX/'
solution_dir='curve_pose_opt2/'
data_dir=dataset+solution_dir

###reference frame transformation
with open(data_dir+'curve_pose.yaml') as file:
	curve_pose = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

print(curve_pose[:-1,-1])
print(np.degrees(rotationMatrixToEulerAngles(curve_pose[:3,:3])))

