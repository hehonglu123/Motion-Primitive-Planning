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


data_dir='wood/dual_arm/'
num_ls=[50]
robot=abb6640(d=50)
curve_js = read_csv(data_dir+'arm1.csv',header=None).values


curve = []
for q in curve_js:
	curve.append(robot.fwd(q).p)
curve=np.array(curve)


curve_fit=np.zeros((len(curve_js),3))
curve_fit_R=np.zeros((len(curve_js),3,3))
for num_l in num_ls:
	breakpoints=np.linspace(0,len(curve_js),num_l+1).astype(int)
	points=[]
	points.append([np.array(curve[0,:3])])
	q_bp=[]
	q_bp.append([np.array(curve_js[0])])
	for i in range(1,num_l+1):
		points.append([np.array(curve_js[breakpoints[i]-1])])
		q_bp.append([np.array(curve_js[breakpoints[i]-1])])
		
	primitives_choices=['movej_fit']*(num_l+1)

	df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices,'points':points,'q_bp':q_bp})
	df.to_csv(data_dir+'command1.csv',header=True,index=False)


with open(data_dir+'tcp.yaml') as file:
    H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

robot=abb1200(R_tool=H_tcp[:3,:3],p_tool=H_tcp[:-1,-1])

curve_js = read_csv(data_dir+'arm2.csv',header=None).values

curve = []
for q in curve_js:
	curve.append(robot.fwd(q).p)
curve=np.array(curve)


curve_fit=np.zeros((len(curve_js),3))
curve_fit_R=np.zeros((len(curve_js),3,3))
for num_l in num_ls:
	breakpoints=np.linspace(0,len(curve_js),num_l+1).astype(int)
	points=[]
	points.append([np.array(curve[0,:3])])
	q_bp=[]
	q_bp.append([np.array(curve_js[0])])
	for i in range(1,num_l+1):
		points.append([np.array(curve_js[breakpoints[i]-1])])
		q_bp.append([np.array(curve_js[breakpoints[i]-1])])

		
	primitives_choices=['movej_fit']*(num_l+1)

	df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices,'points':points,'q_bp':q_bp})
	df.to_csv(data_dir+'command2.csv',header=True,index=False)
