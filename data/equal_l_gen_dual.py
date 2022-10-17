import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from robots_def import *
from utils import *
from lambda_calc import *

from lambda_calc import *
from utils import *


data_dir='from_NX/dual_arm/'
solution_dir='diffevo_pose2_2/'
cmd_dir=data_dir+solution_dir+'10L/'
num_ls=[10]
robot=abb6640(d=50)
curve_js = read_csv(data_dir+solution_dir+'arm1.csv',header=None).values


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
		points.append([np.array(curve[breakpoints[i]-1,:3])])
		q_bp.append([np.array(curve_js[breakpoints[i]-1])])

		if i==1:
			curve_fit[breakpoints[i-1]:breakpoints[i]]=np.linspace(curve[breakpoints[i-1],:3],curve[breakpoints[i]-1,:3],num=breakpoints[i]-breakpoints[i-1])
			R_init=robot.fwd(curve_js[breakpoints[i-1]]).R
			R_end=robot.fwd(curve_js[breakpoints[i]-1]).R
			curve_fit_R[breakpoints[i-1]:breakpoints[i]]=orientation_interp(R_init,R_end,breakpoints[i]-breakpoints[i-1])

		else:
			curve_fit[breakpoints[i-1]:breakpoints[i]]=np.linspace(curve[breakpoints[i-1]-1,:3],curve[breakpoints[i]-1,:3],num=breakpoints[i]-breakpoints[i-1]+1)[1:]
			R_init=robot.fwd(curve_js[breakpoints[i-1]-1]).R
			R_end=robot.fwd(curve_js[breakpoints[i]-1]).R
			curve_fit_R[breakpoints[i-1]:breakpoints[i]]=orientation_interp(R_init,R_end,breakpoints[i]-breakpoints[i-1]+1)[1:]
		
	primitives_choices=['moveabsj']+['movel']*num_l
	breakpoints[1:]=breakpoints[1:]-1
	df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices,'p_bp':points,'q_bp':q_bp})
	df.to_csv(cmd_dir+'command1.csv',header=True,index=False)

	# curve_fit_js=car2js(robot,curve_js[0],curve_fit,curve_fit_R)
	# DataFrame(curve_fit_js).to_csv(data_dir+'curve_fit_js1.csv',header=False,index=False)
	# DataFrame(curve_fit).to_csv(data_dir+'curve_fit1.csv',header=False,index=False)



with open(data_dir+'tcp.yaml') as file:
    H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

robot=abb1200(R_tool=H_tcp[:3,:3],p_tool=H_tcp[:-1,-1])

curve_js = read_csv(data_dir+solution_dir+'arm2.csv',header=None).values

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
		points.append([np.array(curve[breakpoints[i]-1,:3])])
		q_bp.append([np.array(curve_js[breakpoints[i]-1])])

		if i==1:
			curve_fit[breakpoints[i-1]:breakpoints[i]]=np.linspace(curve[breakpoints[i-1],:3],curve[breakpoints[i]-1,:3],num=breakpoints[i]-breakpoints[i-1])
			R_init=robot.fwd(curve_js[breakpoints[i-1]]).R
			R_end=robot.fwd(curve_js[breakpoints[i]-1]).R
			curve_fit_R[breakpoints[i-1]:breakpoints[i]]=orientation_interp(R_init,R_end,breakpoints[i]-breakpoints[i-1])

		else:
			curve_fit[breakpoints[i-1]:breakpoints[i]]=np.linspace(curve[breakpoints[i-1]-1,:3],curve[breakpoints[i]-1,:3],num=breakpoints[i]-breakpoints[i-1]+1)[1:]
			R_init=robot.fwd(curve_js[breakpoints[i-1]-1]).R
			R_end=robot.fwd(curve_js[breakpoints[i]-1]).R
			curve_fit_R[breakpoints[i-1]:breakpoints[i]]=orientation_interp(R_init,R_end,breakpoints[i]-breakpoints[i-1]+1)[1:]
		
	primitives_choices=['moveabsj']+['moveabsj']*num_l
	breakpoints[1:]=breakpoints[1:]-1
	df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices,'p_bp':points,'q_bp':q_bp})
	df.to_csv(cmd_dir+'command2.csv',header=True,index=False)

	# curve_fit_js=car2js(robot,curve_js[0],curve_fit,curve_fit_R)
	# DataFrame(curve_fit_js).to_csv(data_dir+'curve_fit_js2.csv',header=False,index=False)
	# DataFrame(curve_fit).to_csv(data_dir+'curve_fit2.csv',header=False,index=False)