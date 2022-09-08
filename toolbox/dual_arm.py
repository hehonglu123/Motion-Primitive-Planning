import numpy as np
import copy
from general_robotics_toolbox import *
from utils import *
from lambda_calc import *


def form_relative_path(robot1,robot2,curve_js1,curve_js2,base2_R,base2_p):

	relative_path_exe=[]
	relative_path_exe_R=[]
	curve_exe1=[]
	curve_exe2=[]
	curve_exe_R1=[]
	curve_exe_R2=[]
	for i in range(len(curve_js1)):
		pose1_now=robot1.fwd(curve_js1[i])
		pose2_now=robot2.fwd(curve_js2[i])

		curve_exe1.append(pose1_now.p)
		curve_exe2.append(pose2_now.p)
		curve_exe_R1.append(pose1_now.R)
		curve_exe_R2.append(pose2_now.R)

		pose2_world_now=robot2.fwd(curve_js2[i],base2_R,base2_p)


		relative_path_exe.append(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p))
		relative_path_exe_R.append(pose2_world_now.R.T@pose1_now.R)
	return np.array(relative_path_exe),np.array(relative_path_exe_R)

def calc_individual_speed(vd_relative,lam1,lam2,lam_relative,breakpoints):
	speed_ratio=[]      ###speed of robot1 TCP / robot2 TCP
	dt=[]
	for i in range(1,len(breakpoints)):
		speed_ratio.append((lam1[breakpoints[i]-1]-lam1[breakpoints[i-1]-1])/(lam2[breakpoints[i]-1]-lam2[breakpoints[i-1]-1]))
		dt.append((lam_relative[breakpoints[i]]-lam_relative[breakpoints[i-1]])/vd_relative)
	###specify speed here for robot2
	s2_all=[200]
	s1_all=[200]
	for i in range(len(breakpoints)-1):

		s2_all.append((lam2[breakpoints[i+1]]-lam2[breakpoints[i]])/dt[i])
		s1=s2_all[i+1]*speed_ratio[i]
		s1_all.append(s1)

	return s1_all,s2_all

def initialize_data(dataset,data_dir,solution_dir,cmd_dir):
	relative_path = read_csv(data_dir+"Curve_dense.csv", header=None).values

	###calculate desired TCP speed for both arm
	curve_js1=read_csv(solution_dir+'arm1.csv', header=None).values
	curve_js2=read_csv(solution_dir+'arm2.csv', header=None).values

	

	lam_relative_path=calc_lam_cs(relative_path)

	with open(solution_dir+'abb1200.yaml') as file:
		H_1200 = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

	base2_R=H_1200[:3,:3]
	base2_p=1000*H_1200[:-1,-1]

	with open(solution_dir+'tcp.yaml') as file:
		H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
	robot1=abb6640(d=50)
	robot2=abb1200(R_tool=H_tcp[:3,:3],p_tool=H_tcp[:-1,-1])

	lam1=calc_lam_js(curve_js1,robot1)
	lam2=calc_lam_js(curve_js2,robot2)


	return relative_path,robot1,robot2,base2_R,base2_p,lam_relative_path,lam1,lam2,curve_js1,curve_js2