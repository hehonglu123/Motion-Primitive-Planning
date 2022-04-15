import numpy as np
import copy
from general_robotics_toolbox import *
from utils import *
###calculate distance between point to line
def get_distance(p1,p2,p3):
	return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)

###calculate maximum error of 1 point and the curve in cartesian space, distance only
# def calc_error(p,curve):
# 	dist=np.linalg.norm(curve-np.tile(p,(len(curve),1)),axis=1)
# 	order=np.argsort(dist)
# 	error=get_distance(curve[order[0]],curve[order[1]],p)
# 	return error
def calc_error(p,curve):
	dist=np.linalg.norm(curve-np.tile(p,(len(curve),1)),axis=1)
	order=np.argsort(dist)
	return dist[order[0]], order[0]
	
###calculate maximum error between fit curve and original curve in cartesian space, distance only
def calc_max_error(fit,curve):
	max_error=0
	idx=0
	max_error_idx=0
	for p in fit:
		error,idx2=calc_error(p,curve)
		if error>max_error:
			max_error_idx=idx
			max_error=copy.deepcopy(error)
		idx+=1

	return max_error, max_error_idx

def calc_max_error_w_normal(fit,curve,fit_normal,curve_normal):
	max_error=0
	max_error_angle=0
	idx=0
	max_error_idx=0
	for i in range(len(fit)):
		error,idx2=calc_error(fit[i],curve)
		normal_angle=get_angle(fit_normal[i],curve_normal[idx2])
		if error>max_error:
			max_error_idx=idx
			max_error=copy.deepcopy(error)
		if normal_angle>max_error_angle:
			max_error_angle=copy.deepcopy(normal_angle)
		idx+=1

	return max_error,max_error_angle, max_error_idx

def calc_max_error_js(robot,fit_js,curve_js):
	fit=[]
	for i in range(len(fit_js)):
		fit.append(robot.fwd(fit_js[i]).p)
	curve=[]
	for i in range(len(curve_js)):
		curve.append(robot.fwd(curve_js[i]).p)
		

	max_error=0
	idx=0
	max_error_idx=0
	for p in fit:
		error=calc_error(p,curve)
		if error>max_error:
			max_error_idx=idx
			max_error=copy.deepcopy(error)
		idx+=1

	return max_error, max_error_idx

def calc_avg_error(fit,curve):
	error=calc_all_error(fit,curve)
	return sum(error)/len(error)

def calc_all_error(fit,curve):
	error=[]
	for p in fit:
		error_temp,idx=calc_error(p,curve)
		error.append(error_temp)
	return error

def complete_points_check(fit,curve,R_fit,R_curve):
	error=[]
	
	rotation_error=[]
	for i in range(len(fit)):
		error_temp=np.linalg.norm(curve-fit[i],axis=1)
		idx=np.argmin(error_temp)
		error.append(error_temp[idx])

		R=np.dot(R_fit[i],R_curve[i].T)
		k,theta=R2rot(R)
		rotation_error.append(theta)

	error=np.array(error)
	max_cartesian_error=np.max(error)
	avg_cartesian_error=np.average(error)
	max_cartesian_error_index=np.argmax(error)

	return max_cartesian_error,max_cartesian_error_index,avg_cartesian_error,np.max(np.array(rotation_error))

def complete_points_check2(fit_backproj,curve_backproj,fit,curve):	###error metric on 9/17 by prof Julius
	error_backproj=[]
	
	error=[]
	for i in range(len(fit)):
		error_temp=np.linalg.norm(curve-fit[i],axis=1)
		idx=np.argmin(error_temp)
		error.append(error_temp[idx])

		error_temp=np.linalg.norm(curve_backproj-fit_backproj[i],axis=1)
		idx=np.argmin(error_temp)
		error_backproj.append(error_temp[idx])

	error=np.array(error)
	error_backproj=np.array(error_backproj)
	error_total=error+error_backproj

	max_cartesian_error=np.max(error)
	avg_cartesian_error=np.average(error)
	max_cartesian_error_backproj=np.max(error_backproj)
	avg_cartesian_error_backproj=np.average(error_backproj)
	max_total_error=np.max(error_total)
	avg_total_error=np.average(error_total)
	max_error_index=np.argmax(error_total)

	return max_cartesian_error,avg_cartesian_error,max_cartesian_error_backproj,avg_cartesian_error_backproj,max_total_error,avg_total_error,max_error_index

