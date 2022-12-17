import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys

sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from utils import *
from lambda_calc import *


dataset='curve_2'
curve = read_csv("../data/"+dataset+'/baseline/Curve_in_base_frame.csv',header=None).values


data = np.loadtxt(dataset+'_base_frame/end_waypoints.csv',delimiter=',', skiprows=1,usecols = (0,1,2,3,4,5,6))

curve_exe=data[:,1:4]*1000
error=calc_all_error(curve_exe,curve[:,:3])

lam=calc_lam_cs(curve_exe)

plt.plot(lam,error)
plt.show()