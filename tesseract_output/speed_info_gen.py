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
# curve = read_csv("../data/"+dataset+'/baseline/Curve_in_base_frame.csv',header=None).values
curve_dense=read_csv("../data/"+dataset+'/Curve_dense.csv',header=None).values

# data = np.loadtxt(dataset+'_base_frame/end_waypoints.csv',delimiter=',', skiprows=1,usecols = (0,1,2,3,4,5,6))
data = np.loadtxt(dataset+'_dense/end_waypoints.csv',delimiter=',', skiprows=1,usecols = (0,1,2,3,4,5,6))

H=H_from_RT(Rz(np.pi/2),[1700,-1000,1400])

curve_exe=data[:,1:4]*1000
curve_rotated=[]
curve_rotated=np.dot(H[:3,:3],curve_dense[:,:3].T).T+np.tile(H[:-1,-1],(len(curve_dense),1))


error=calc_all_error(curve_exe,curve_rotated)

lam=calc_lam_cs(curve_exe)

plt.plot(lam,error)
plt.show()