import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('../toolbox')
from robots_def import *
from utils import *
from lambda_calc import *
from MotionSend import *

robot=abb6640(d=50)
#determine correct commanded speed to keep error within 1mm
thresholds=[0.1,0.2,0.5,0.9]

dataset="data/from_NX/"
curve = read_csv(dataset+"Curve_in_base_frame.csv",header=None).values
for threshold in thresholds:
    ms = MotionSend()
    data_dir=dataset+"baseline/"+str(threshold)+'/'

    

