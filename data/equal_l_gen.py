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

sys.path.append('../../toolbox')
from lambda_calc import *
from utils import *


data_dir='wood/'
num_ls=[80,100,150]
robot=abb6640(d=50)
curve_js = read_csv(data_dir+'Curve_js.csv',header=None).values
curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
for num_l in num_ls:
	breakpoints=np.linspace(0,len(curve_js),num_l+1).astype(int)
	points=[]
	points.append([np.array(curve_js[0])])
	for i in range(1,num_l+1):
		points.append([np.array(curve[breakpoints[i]-1,:3])])

	primitives_choices=['movej_fit']+['movel_fit']*num_l

	df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices,'points':points})
	df.to_csv(data_dir+'baseline/'+str(num_l)+'L/command.csv',header=True,index=False)

