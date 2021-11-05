import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from pandas import *
import sys
sys.path.append('../circular_fit')
from toolbox_circular_fit import *
sys.path.append('../toolbox')
from robot_def import *

#####################3d curve-fitting with MoveL, MoveJ, MoveC; stepwise incremental bi-section searched breakpoints###############################

def project(curve_fit,R_init,R_last):
	d=50
	###############interpolate orientation linearly#################
	###calculate total distance
	total_dis=np.sum(np.linalg.norm(curve_fit[1:]-curve_fit[:-1],axis=1))
	distance_traveled=0
	curve_fit_proj=[]
	###find axis angle first
	R_diff=np.dot(R_init.T,R_last)
	k,theta=R2rot(R_diff)

	for i in range(len(curve_fit)):
		distance_traveled+=np.linalg.norm(curve_fit[i]-curve_fit[max(0,i-1)])
		###linearly interpolate angle
		angle=theta*distance_traveled/total_dis
		R=rot(k,angle)
		R_act=np.dot(R_init,R)
		curve_fit_proj.append(curve_fit[i]+d*R_act[:,-1])

	return curve_fit_proj

def movel_fit(curve,curve_backproj,curve_backproj_js,q=[]):
	###no constraint
	if len(q)==0:
		A=np.vstack((np.ones(len(curve_backproj)),np.arange(0,len(curve_backproj)))).T
		b=curve_backproj
		res=np.linalg.lstsq(A,b,rcond=None)[0]
		start_point=res[0]
		slope=res[1].reshape(1,-1)

		start_pose=fwd(curve_backproj_js[0])
	###with constraint point
	else:
		start_pose=fwd(q)
		p=start_pose.p
		A=np.arange(0,len(curve_backproj)).reshape(-1,1)
		b=curve_backproj-curve_backproj[0]
		res=np.linalg.lstsq(A,b,rcond=None)[0]
		start_point=curve_backproj[0]
		slope=res.reshape(1,-1)

	curve_fit=np.dot(np.arange(0,len(curve_backproj)).reshape(-1,1),slope)+start_point
	###calculate fitting error
	max_error1=np.max(np.linalg.norm(curve_backproj-curve_fit,axis=1))

	

	end_R=fwd(curve_backproj_js[-1]).R
	q_all=np.array(inv(curve_fit[-1],end_R))

	###choose inv_kin closest to previous joints
	temp_q=q_all-curve_backproj_js[-1]
	order=np.argsort(np.linalg.norm(temp_q,axis=1))
	q_last=q_all[order[0]]

	###calculating projection error
	curve_proj=project(curve_fit,start_pose.R,end_R)
	max_error2=np.max(np.linalg.norm(curve-curve_proj,axis=1))
	max_error=(max_error1+max_error2)/2

	# print(max_error1,max_error2)
	return curve_fit,q_last,max_error


def movej_fit(curve,curve_backproj,curve_backproj_js,q=[]):
	if len(q)==0:
		A=np.vstack((np.ones(len(curve_backproj_js)),np.arange(0,len(curve_backproj_js)))).T
		b=curve_backproj_js
		res=np.linalg.lstsq(A,b,rcond=None)[0]
		start_point=res[0]
		slope=res[1].reshape(1,-1)

		start_pose=fwd(curve_backproj_js[0])
	###with constraint point
	else:
		start_pose=fwd(q)
		A=np.arange(0,len(curve_backproj_js)).reshape(-1,1)
		b=curve_backproj_js-curve_backproj_js[0]
		res=np.linalg.lstsq(A,b,rcond=None)[0]
		start_point=curve_backproj_js[0]
		slope=res.reshape(1,-1)

	curve_js_fit=np.dot(np.arange(0,len(curve_backproj_js)).reshape(-1,1),slope)+start_point

	curve_fit=[]
	for i in range(len(curve_js_fit)):
		curve_fit.append(fwd(curve_js_fit[i]).p)
	curve_fit=np.array(curve_fit)

	###calculate fitting error
	max_error1=np.max(np.linalg.norm(curve_backproj-curve_fit,axis=1))

	###calculating projection error
	curve_proj=project(curve_fit,start_pose.R,fwd(curve_js_fit[-1]).R)
	max_error2=np.max(np.linalg.norm(curve-curve_proj,axis=1))
	max_error=(max_error1+max_error2)/2

	return curve_fit,curve_js_fit[-1],max_error


def movec_fit(curve,curve_backproj,curve_backproj_js,q=[]):
	if len(q)==0:
		p=[]
		start_pose=fwd(curve_backproj_js[0])
	###with constraint point
	else:
		start_pose=fwd(q)
		p=fwd(q).p
	
	curve_fit,curve_fit_circle=circle_fit(curve_backproj,p)
	max_error1=np.max(np.linalg.norm(curve_backproj-curve_fit,axis=1))

	end_R=fwd(curve_backproj_js[-1]).R
	q_all=np.array(inv(curve_fit[-1],end_R))

	###choose inv_kin closest to previous joints
	temp_q=q_all-curve_backproj_js[-1]
	order=np.argsort(np.linalg.norm(temp_q,axis=1))
	q_last=q_all[order[0]]


	###calculating projection error

	curve_proj=project(curve_fit,start_pose.R,end_R)
	max_error2=np.max(np.linalg.norm(curve-curve_proj,axis=1))
	max_error=(max_error1+max_error2)/2

	return curve_fit,q_last,max_error


def fit_under_error(curve,curve_backproj,curve_backproj_js,max_error_threshold,d=50):

	###initialize
	breakpoints=[0]
	primitives_choices=[]
	q_breakpoints=[]
	points=[]


	results_max_cartesian_error=[]
	results_max_cartesian_error_index=[]
	results_avg_cartesian_error=[]
	results_max_orientation_error=[]
	results_max_dz_error=[]
	results_avg_dz_error=[]

	fit=[]
	js_fit=[]

	while breakpoints[-1]!=len(curve)-1:
		###primitive candidates
		primitives={'movel_fit':movel_fit,'movej_fit':movej_fit,'movec_fit':movec_fit}
		
		

		next_point= min(2000,len(curve)-1-breakpoints[-1])
		prev_point=0
		prev_possible_point=0

		max_errors={'movel_fit':999,'movej_fit':999,'movec_fit':999}

		###bisection search breakpoints
		while True:
			###bp going backward to meet threshold
			if min(list(max_errors.values()))>max_error_threshold:
				prev_point_temp=next_point
				next_point-=int(np.abs(next_point-prev_point)/2)
				prev_point=prev_point_temp
				
				for key in primitives: 
					curve_fit,q_last,max_error=primitives[key](curve[breakpoints[-1]:breakpoints[-1]+next_point],curve_backproj[breakpoints[-1]:breakpoints[-1]+next_point],curve_backproj_js[breakpoints[-1]:breakpoints[-1]+next_point],q=[] if len(q_breakpoints)==0 else q_breakpoints[-1])
					max_errors[key]=max_error



			###bp going forward to get close to threshold
			else:
				prev_possible_point=next_point
				prev_point_temp=next_point
				next_point= min(next_point + int(np.abs(next_point-prev_point)),len(curve)-1-breakpoints[-1])
				prev_point=prev_point_temp
				

				for key in primitives: 
					curve_fit,q_last,max_error=primitives[key](curve[breakpoints[-1]:breakpoints[-1]+next_point],curve_backproj[breakpoints[-1]:breakpoints[-1]+next_point],curve_backproj_js[breakpoints[-1]:breakpoints[-1]+next_point],q=[] if len(q_breakpoints)==0 else q_breakpoints[-1])
					max_errors[key]=max_error

			# print(max_errors)
			if next_point==prev_point:
				print('stuck')
				###if ever getting stuck, restore
				next_point=prev_possible_point
				
				
				for key in primitives: 
					curve_fit,q_last,max_error=primitives[key](curve[breakpoints[-1]:breakpoints[-1]+next_point],curve_backproj[breakpoints[-1]:breakpoints[-1]+next_point],curve_backproj_js[breakpoints[-1]:breakpoints[-1]+next_point],q=[] if len(q_breakpoints)==0 else q_breakpoints[-1])
					if max_error<max_error_threshold:
						q_breakpoints.append(q_last)
						primitives_choices.append(key)
						if key=='movec_fit':
							points.append([curve_fit[int(len(curve_fit)/2)],curve_fit[-1]])
						elif key=='movel_fit':
							points.append([curve_fit[-1]])
						else:
							points.append([q_last])
						break

				

				breakpoints.append(breakpoints[-1]+next_point)
				fit.append(curve_fit)
				break

			###find the closest but under max_threshold
			if (min(list(max_errors.values()))<=max_error_threshold and np.abs(next_point-prev_point)<10) or next_point==len(curve)-1:
				for key in primitives: 
					curve_fit,q_last,max_error=primitives[key](curve[breakpoints[-1]:breakpoints[-1]+next_point],curve_backproj[breakpoints[-1]:breakpoints[-1]+next_point],curve_backproj_js[breakpoints[-1]:breakpoints[-1]+next_point],q=[] if len(q_breakpoints)==0 else q_breakpoints[-1])
					if max_error<max_error_threshold:
						q_breakpoints.append(q_last)
						primitives_choices.append(key)
						if key=='movec_fit':
							points.append([curve_fit[int(len(curve_fit)/2)],curve_fit[-1]])
						elif key=='movel_fit':
							points.append([curve_fit[-1]])
						else:
							points.append([q_last])
						break

				breakpoints.append(breakpoints[-1]+next_point)
				fit.append(curve_fit)


				break


		print(breakpoints)
		print(primitives_choices)
		# print(points)

	##############################check error (against fitting forward projected curve)##############################
	fit_all=np.vstack(fit)
	error=[]
	for i in range(len(fit_all)):
	    error_temp=np.linalg.norm(curve_backproj-fit_all[i],axis=1)
	    idx=np.argmin(error_temp)
	    error.append(error_temp[idx])

	error=np.array(error)
	max_error=np.max(error)
	print('max error: ', max_error)


	###plt
	###3D plot
	plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'gray')
	for i in range(len(fit)):
		ax.scatter3D(fit[i][:,0], fit[i][:,1], fit[i][:,2], c=fit[i][:,2], cmap='Greens')
	plt.show()

	return breakpoints,primitives_choices,points,fit_all




def main():
	###read in points
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../data/from_cad/Curve_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T

	###read in points backprojected
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../data/from_cad/Curve_backproj_in_base_frame.csv", names=col_names)
	curve_backproj_x=data['X'].tolist()
	curve_backproj_y=data['Y'].tolist()
	curve_backproj_z=data['Z'].tolist()
	curve_backproj=np.vstack((curve_backproj_x, curve_backproj_y, curve_backproj_z)).T

	###read interpolated curves in joint space
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("../data/from_cad/Curve_js.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	###read interpolated curves in joint space
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("../data/from_cad/Curve_backproj_js.csv", names=col_names)
	curve_backproj_q1=data['q1'].tolist()
	curve_backproj_q2=data['q2'].tolist()
	curve_backproj_q3=data['q3'].tolist()
	curve_backproj_q4=data['q4'].tolist()
	curve_backproj_q5=data['q5'].tolist()
	curve_backproj_q6=data['q6'].tolist()
	curve_backproj_js=np.vstack((curve_backproj_q1, curve_backproj_q2, curve_backproj_q3,curve_backproj_q4,curve_backproj_q5,curve_backproj_q6)).T

	breakpoints,primitives_choices,points,curve_fit=fit_under_error(curve,curve_backproj,curve_backproj_js,1.)

	###insert initial configuration
	primitives_choices.insert(0,'movej_fit')

	q_all=np.array(inv(curve_fit[0],fwd(curve_backproj_js[0]).R))

	###choose inv_kin closest to previous joints
	temp_q=q_all-curve_backproj_js[0]
	order=np.argsort(np.linalg.norm(temp_q,axis=1))
	q_init=q_all[order[0]]


	points.insert(0,[q_init])
	print(primitives_choices)
	print(points)

	df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices,'points':points})
	df.to_csv('comparison/moveL+moveC/command_backproj.csv',header=True,index=False)
	df=DataFrame({'x':curve_fit[:,0],'y':curve_fit[:,1],'z':curve_fit[:,2]})
	df.to_csv('comparison/moveL+moveC/curve_fit_backproj.csv',header=True,index=False)

if __name__ == "__main__":
	main()
