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
solution_dir='curve_pose_opt3/'

num_js=[100]
robot=abb6640(d=50)
# curve_js = read_csv(data_dir+'Curve_js.csv',header=None).values
curve_js = read_csv(data_dir+solution_dir+'Curve_js.csv',header=None).values
curve = read_csv(data_dir+solution_dir+"Curve_in_base_frame.csv",header=None).values

curve_fit=[]
curve_fit_js=np.zeros((len(curve_js),6))
for num_j in num_js:
	cmd_dir=data_dir+solution_dir+str(num_j)+'J/'

	breakpoints=np.linspace(0,len(curve_js),num_j+1).astype(int)
	points=[]
	points.append([np.array(curve[0,:3])])
	q_bp=[]
	q_bp.append([np.array(curve_js[0])])
	for i in range(1,num_j+1):
		points.append([np.array(curve[breakpoints[i]-1,:3])])
		q_bp.append([np.array(curve_js[breakpoints[i]-1])])

		if i==1:
			curve_fit_js[breakpoints[i-1]:breakpoints[i]]=np.linspace(curve_js[breakpoints[i-1]],curve_js[breakpoints[i]-1],num=breakpoints[i]-breakpoints[i-1])

		else:
			curve_fit_js[breakpoints[i-1]:breakpoints[i]]=np.linspace(curve_js[breakpoints[i-1]-1],curve_js[breakpoints[i]-1],num=breakpoints[i]-breakpoints[i-1]+1)[1:]
		
	primitives_choices=['movej_fit']*(num_j+1)

	df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices,'points':points,'q_bp':q_bp})
	df.to_csv(cmd_dir+'command.csv',header=True,index=False)

	for i in range(len(curve_fit_js)):
		curve_fit.append(robot.fwd(curve_fit_js[i]).p)
	curve_fit=np.array(curve_fit)
	DataFrame(curve_fit_js).to_csv(cmd_dir+'curve_fit_js.csv',header=False,index=False)
	DataFrame(curve_fit).to_csv(cmd_dir+'/curve_fit.csv',header=False,index=False)


