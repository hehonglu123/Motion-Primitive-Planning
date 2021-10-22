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

def movel_fit(curve,curve_js,q=[]):
	###no constraint
	if len(q)==0:
		A=np.vstack((np.ones(len(curve)),np.arange(0,len(curve)))).T
		b=curve
		res=np.linalg.lstsq(A,b,rcond=None)[0]
		start_point=res[0]
		slope=res[1].reshape(1,-1)
	###with constraint point
	else:
		p=fwd(q).p
		A=np.arange(0,len(curve)).reshape(-1,1)
		b=curve-curve[0]
		res=np.linalg.lstsq(A,b,rcond=None)[0]
		start_point=curve[0]
		slope=res.reshape(1,-1)

	curve_fit=np.dot(np.arange(0,len(curve)).reshape(-1,1),slope)+start_point

	max_error=np.max(np.linalg.norm(curve-curve_fit,axis=1))

	q_all=np.array(inv(curve_fit[-1],fwd(curve_js[-1]).R))

	###choose inv_kin closest to previous joints
	temp_q=q_all-curve_js[-1]
	order=np.argsort(np.linalg.norm(temp_q,axis=1))
	q_last=q_all[order[0]]

	return curve_fit,q_last,max_error


def movej_fit(curve,curve_js,q=[]):
	if len(q)==0:
		A=np.vstack((np.ones(len(curve_js)),np.arange(0,len(curve_js)))).T
		b=curve_js
		res=np.linalg.lstsq(A,b,rcond=None)[0]
		start_point=res[0]
		slope=res[1].reshape(1,-1)
	else:
		A=np.arange(0,len(curve_js)).reshape(-1,1)
		b=curve_js-curve_js[0]
		res=np.linalg.lstsq(A,b,rcond=None)[0]
		start_point=curve_js[0]
		slope=res.reshape(1,-1)

	curve_js_fit=np.dot(np.arange(0,len(curve_js)).reshape(-1,1),slope)+start_point

	curve_fit=[]
	for i in range(len(curve_js_fit)):
		curve_fit.append(fwd(curve_js_fit[i]).p)
	curve_fit=np.array(curve_fit)
	max_error=np.max(np.linalg.norm(curve-curve_fit,axis=1))

	return curve_fit,curve_js_fit[-1],max_error



def movec_fit(curve,curve_js,q=[]):
	if len(q)>0:
		p=fwd(q).p
	else:
		p=[]
	curve_fit,max_error=seg_3dfit(curve,p)

	q_all=np.array(inv(curve_fit[-1],fwd(curve_js[-1]).R))

	###choose inv_kin closest to previous joints
	temp_q=q_all-curve_js[-1]
	order=np.argsort(np.linalg.norm(temp_q,axis=1))
	q_last=q_all[order[0]]

	return curve_fit,q_last,max_error


def fit_under_error(curve,curve_js,max_error_threshold,d=50):

	###initialize
	breakpoints=[0]
	num_breakpoints=4
	primitives_choices=[]
	q_breakpoints=[]
	primitives_data=[]


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
					curve_fit,q_last,max_error=primitives[key](curve[breakpoints[-1]:breakpoints[-1]+next_point],curve_js[breakpoints[-1]:breakpoints[-1]+next_point],q=[] if len(q_breakpoints)==0 else q_breakpoints[-1])
					max_errors[key]=max_error



			###bp going forward to get close to threshold
			else:
				prev_possible_point=next_point
				prev_point_temp=next_point
				next_point= min(next_point + int(np.abs(next_point-prev_point)),len(curve)-1-breakpoints[-1])
				prev_point=prev_point_temp
				

				for key in primitives: 
					curve_fit,q_last,max_error=primitives[key](curve[breakpoints[-1]:breakpoints[-1]+next_point],curve_js[breakpoints[-1]:breakpoints[-1]+next_point],q=[] if len(q_breakpoints)==0 else q_breakpoints[-1])
					max_errors[key]=max_error

			# print(max_errors)
			if next_point==prev_point:
				print('stuck')
				###if ever getting stuck, restore
				next_point=prev_possible_point
				
				
				for key in primitives: 
					curve_fit,q_last,max_error=primitives[key](curve[breakpoints[-1]:breakpoints[-1]+next_point],curve_js[breakpoints[-1]:breakpoints[-1]+next_point],q=[] if len(q_breakpoints)==0 else q_breakpoints[-1])
					if max_error<max_error_threshold:
						q_breakpoints.append(q_last)
						primitives_choices.append(key)
						if key=='movec_fit':
							primitives_data.append([curve_fit[int(len(curve_fit)/2)],curve_fit[-1]])
						elif key=='movel_fit':
							primitives_data.append([curve_fit[-1]])
						else:
							primitives_data.append([q_last])
						break

				

				breakpoints.append(breakpoints[-1]+next_point)
				fit.append(curve_fit)
				break

			###find the closest but under max_threshold
			if (min(list(max_errors.values()))<=max_error_threshold and np.abs(next_point-prev_point)<10) or next_point==len(curve)-1:
				for key in primitives: 
					curve_fit,q_last,max_error=primitives[key](curve[breakpoints[-1]:breakpoints[-1]+next_point],curve_js[breakpoints[-1]:breakpoints[-1]+next_point],q=[] if len(q_breakpoints)==0 else q_breakpoints[-1])
					if max_error<max_error_threshold:
						q_breakpoints.append(q_last)
						primitives_choices.append(key)
						if key=='movec_fit':
							primitives_data.append([curve_fit[int(len(curve_fit)/2)],curve_fit[-1]])
						elif key=='movel_fit':
							primitives_data.append([curve_fit[-1]])
						else:
							primitives_data.append([q_last])
						break

				breakpoints.append(breakpoints[-1]+next_point)
				fit.append(curve_fit)


				break


		print(breakpoints)
		print(primitives_choices)
		print(primitives_data)

	##############################check error (against fitting forward projected curve)##############################
	fit_all=np.vstack(fit)
	error=[]
	for i in range(len(fit_all)):
	    error_temp=np.linalg.norm(curve-fit_all[i],axis=1)
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

	return breakpoints,primitives_choices,primitives_data




def main():
	###read in points
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../data/from_cad/Curve_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T

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

	breakpoints,primitives_choices,primitives_data=fit_under_error(curve,curve_js,1.)
	print(len(breakpoints[1:]),len(primitives_choices),len(primitives_data))
	df=DataFrame({'breakpoints':breakpoints[1:],'primitives_choices':primitives_choices,'primitives_data':primitives_data})
	df.to_csv('command.csv',header=False,index=False)

if __name__ == "__main__":
	main()
