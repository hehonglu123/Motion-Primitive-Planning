import numpy as np
from matplotlib.pyplot import *

from pandas import *
from fitting_toolbox_new import *
import sys
sys.path.append('../circular_fit')
from toolbox_circular_fit import *
sys.path.append('../toolbox')
from robots_def import *
from direction2R import *
from general_robotics_toolbox import *
from error_check import *

def get_angle(v1,v2,less90=False):
		v1=v1/np.linalg.norm(v1)
		v2=v2/np.linalg.norm(v2)
		dot=np.dot(v1,v2)
		if abs(dot)>0.999999:
			return 0
		angle=np.arccos(dot)
		if less90 and angle>np.pi/2:
			angle=np.pi-angle
		return angle

def orientation_interp(R_init,R_end,steps):
		curve_fit_R=[]
		###find axis angle first
		R_diff=np.dot(R_init.T,R_end)
		k,theta=R2rot(R_diff)
		for i in range(steps):
			###linearly interpolate angle
			angle=theta*i/(steps-1)
			R=rot(k,angle)
			curve_fit_R.append(np.dot(R_init,R))
		curve_fit_R=np.array(curve_fit_R)
		return curve_fit_R

threshold=0.1
slope_threshold=np.radians(30)
def movel_fit(curve,curve_js,prev_slope,next_slope,js_prev=[]):
	curve_R_end=robot.fwd(curve_js[-1]).R

	if len(js_prev)==0:
		start_p=curve[0]
		start_R=robot.fwd(curve_js[0]).R
		curve_fit=np.linspace(start_p,curve[-1],len(curve))
		curve_fit_R=orientation_interp(start_R,curve_R_end,len(curve))
	else:
		pose=robot.fwd(js_prev)
		start_p=pose.p
		start_R=pose.R
		curve_fit=np.linspace(start_p,curve[-1],len(curve)+1)[1:]
		curve_fit_R=orientation_interp(start_R,curve_R_end,len(curve)+1)[1:]
	
	error=np.max(np.linalg.norm(curve_fit-curve,axis=1))

	
	
	###check slope diff in js 
	q1_all=robot.inv(curve_fit[0],curve_fit_R[0])
	temp_q=q1_all-curve_js[0]
	order=np.argsort(np.linalg.norm(temp_q,axis=1))
	q1=q1_all[order[0]]

	qlast_all=robot.inv(curve_fit[-1],curve_fit_R[-1])
	temp_q=qlast_all-curve_js[-1]
	order=np.argsort(np.linalg.norm(temp_q,axis=1))
	qlast=qlast_all[order[0]]


	angle1=0
	angle2=0
	if len(prev_slope)>0:
		angle1=abs(get_angle(prev_slope,qlast-q1,True))
	if len(next_slope)>0:
		angle2=abs(get_angle(next_slope,qlast-q1,True))

	return error if max(angle1,angle2)<slope_threshold else 999
def movej_fit(curve,curve_js,prev_slope,next_slope,js_prev=[]):
	if len(js_prev)==0:
		start_q=curve_js[0]
		curve_fit_js=np.linspace(start_q,curve_js[-1],len(curve_js))
	else:
		start_q=js_prev
		curve_fit_js=np.linspace(start_q,curve_js[-1],len(curve_js)+1)[1:]

	curve_fit=[]

	for i in range(len(curve)):
		curve_fit.append(robot.fwd(curve_fit_js[i]).p)
	error=np.max(np.linalg.norm(curve_fit-curve,axis=1))

	angle1=0
	angle2=0
	if len(prev_slope)>0:
		angle1=abs(get_angle(prev_slope,curve_js[-1]-curve_js[0],True))
	if len(next_slope)>0:
		angle2=abs(get_angle(prev_slope,curve_js[-1]-curve_js[0],True))

	return error if max(angle1,angle2)<slope_threshold else 999

def movec_fit(curve,curve_js,prev_slope,next_slope,js_prev=[]):
	curve_R_end=robot.fwd(curve_js[-1]).R

	if len(js_prev)==0:
		start_p=curve[0]
		start_R=robot.fwd(curve_js[0]).R
		curve_fit,circle=circle_fit(curve,start_p,curve[-1])
		curve_fit_R=orientation_interp(start_R,curve_R_end,len(curve))
	else:
		pose=robot.fwd(js_prev)
		start_p=pose.p
		start_R=pose.R
		curve_fit,circle=circle_fit(curve,start_p,curve[-1])
		curve_fit_R=orientation_interp(start_R,curve_R_end,len(curve)+1)[1:]


	
	error=np.max(np.linalg.norm(curve_fit-curve,axis=1))

	angle1=0
	angle2=0
	###find slope in joint space
	if len(prev_slope)>0:
		q1_all=robot.inv(curve_fit[0],curve_fit_R[0])
		temp_q=q1_all-curve_js[0]
		order=np.argsort(np.linalg.norm(temp_q,axis=1))
		q1=q1_all[order[0]]

		q2_all=robot.inv(curve_fit[1],curve_fit_R[1])
		temp_q=q2_all-curve_js[1]
		order=np.argsort(np.linalg.norm(temp_q,axis=1))
		q2=q2_all[order[0]]

		angle1=abs(get_angle(prev_slope,q2-q1,True))

	if len(next_slope)>0:
		qlast1_all=robot.inv(curve_fit[-1],curve_fit_R[-1])
		temp_q=qlast1_all-curve_js[-1]
		order=np.argsort(np.linalg.norm(temp_q,axis=1))
		qlast1=qlast1_all[order[0]]

		qlast2_all=robot.inv(curve_fit[-2],curve_fit_R[-2])
		temp_q=qlast2_all-curve_js[-2]
		order=np.argsort(np.linalg.norm(temp_q,axis=1))
		qlast2=qlast2_all[order[0]]

		angle2=abs(get_angle(next_slope,qlast1-qlast2,True))


	return error if max(angle1,angle2)<slope_threshold else 999

def bisec(curve,curve_js,primitive_fit):
	max_error_threshold=1##bisection search error
	greedy_length=[]
	for i in range(len(curve)):
		next_point=min(2000,len(curve)-i)
		prev_point=0
		error=999
		prev_possible_point=2

		###get slope
		if i==0:
			prev_slope=[]
		else:
			prev_slope=curve_js[i-1]-curve_js[i-2]


		while True:
			# print(next_point)
			if i==len(curve_file_list)-1:
				next_slope=[]
			else:
				next_slope=curve_js[i+1]-curve_js[i-2]
	

			if error>max_error_threshold:
				#going forward
				prev_point_temp=next_point
				next_point-=int(np.abs(next_point-prev_point)/2)
				prev_point=prev_point_temp
				
				error=primitive_fit(curve[i:i+next_point],curve_js[i:i+next_point],prev_slope,next_slope)
			else:
				#going backward
				prev_possible_point=next_point
				prev_point_temp=next_point
				next_point= min(next_point + int(np.abs(next_point-prev_point)),len(curve)-i)
				prev_point=prev_point_temp
				
				error=primitive_fit(curve[i:i+next_point],curve_js[i:i+next_point],prev_slope,next_slope)

			###terminate conditions
			if next_point==prev_point:
				print('stuck, restoring previous possible index')		###if ever getting stuck, restore
				next_point=max(prev_possible_point,2)
				greedy_length.append(next_point)
				break

			###find the closest but under max_threshold
			if error<=max_error_threshold and np.abs(next_point-prev_point)<10:
				greedy_length.append(next_point)
				break
		print(i,greedy_length[-1])
	return greedy_length

import glob, pickle
robot=abb6640(d=50)
# curve_file_list=sorted(glob.glob("../rl_fit/train_data/base/*.csv"))
# with open("../rl_fit/train_data/curve_file_list",'wb') as fp:
# 	pickle.dump(curve_file_list,fp)
with open("../rl_fit/train_data/curve_file_list",'rb') as fp:
	curve_file_list=pickle.load(fp)

for i in range(len(curve_file_list)):
	curve_js_file='../rl_fit/train_data/js_new/'+curve_file_list[i][20:-4]+'_js_new.csv'
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv(curve_file_list[i], names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_normal=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T


	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv(curve_js_file, names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	bisec(curve,curve_js,movec_fit)
	break

