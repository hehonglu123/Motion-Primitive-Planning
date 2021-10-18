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

def movel_fit(curve,p=[]):
	###no constraint
	if len(p)==0:
		A=np.vstack((np.ones(len(curve)),np.arange(0,len(curve)))).T
		b=curve
		res=np.linalg.lstsq(A,b,rcond=None)[0]
		start_point=res[0]
		slope=res[1].reshape(1,-1)
	###with constraint point
	else:
		A=np.arange(0,len(curve)).reshape(-1,1)
		b=curve-curve[0]
		res=np.linalg.lstsq(A,b,rcond=None)[0]
		start_point=curve[0]
		slope=res.reshape(1,-1)

	curve_fit=np.dot(np.arange(0,len(curve)).reshape(-1,1),slope)+start_point

	max_error=np.max(np.linalg.norm(curve-curve_fit,axis=1))

	return curve_fit,max_error


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

	return curve_fit,curve_js_fit,max_error



def movec_fit(curve,p=[]):
	return seg_3dfit(curve,p)

def fit_under_error(curve,curve_js,max_error_threshold,d=50):

	###initialize
	breakpoints=[0]
	num_breakpoints=4
	

	results_max_cartesian_error=[]
	results_max_cartesian_error_index=[]
	results_avg_cartesian_error=[]
	results_max_orientation_error=[]
	results_max_dz_error=[]
	results_avg_dz_error=[]

	fit=[]
	js_fit=[]

	while breakpoints[-1]!=len(curve)-1:
		next_point= min(3000,len(curve)-1-breakpoints[-1])
		increment=100
		###start 1-seg fitting until reaching threshold
		###first test fitting
		# prev_curve_fitarc,max_error=movec_fit(curve[breakpoints[-1]:breakpoints[-1]+next_point],p=[] if breakpoints[-1]>=0 else fit[-1][-1])
		prev_curve_fitarc,prev_curve_js_fitarc,max_error=movej_fit(curve[breakpoints[-1]:breakpoints[-1]+next_point],curve_js[breakpoints[-1]:breakpoints[-1]+next_point],q=[] if breakpoints[-1]>=0 else js_fit[-1][-1])


		if breakpoints[-1]+next_point==len(curve)-1 and max_error<=max_error_threshold:
			breakpoints.append(len(curve)-1)
			fit.append(prev_curve_fitarc)
			js_fit.append(prev_curve_js_fitarc)
			break

		###bp going backward to meet threshold
		if max_error>max_error_threshold:
			while True:
				next_point= min(next_point - increment,len(curve)-1-breakpoints[-1])
				# curve_fitarc,max_error=movec_fit(curve[breakpoints[-1]:breakpoints[-1]+next_point],p=[] if breakpoints[-1]>=0 else fit[-1][-1])
				curve_fitarc,curve_js_fitarc,max_error=movej_fit(curve[breakpoints[-1]:breakpoints[-1]+next_point],curve_js[breakpoints[-1]:breakpoints[-1]+next_point],q=[] if breakpoints[-1]>=0 else js_fit[-1][-1])

				if max_error<=max_error_threshold:
					breakpoints.append(breakpoints[-1]+next_point)
					fit.append(curve_fitarc)
					js_fit.append(curve_js_fitarc)
					break
		###bp going forward to get close to threshold
		else:
			while True:
				next_point= min(next_point + increment,len(curve)-1-breakpoints[-1])
				new_curve_fitarc, max_error=seg_3dfit(curve[breakpoints[-1]:breakpoints[-1]+next_point],p=[] if breakpoints[-1]>=0 else fit[-1][-1])
				if max_error>=max_error_threshold:
					breakpoints.append(breakpoints[-1]+next_point-increment)
					fit.append(prev_curve_fitarc)
					js_fit.append(prev_curve_js_fitarc)
					break
				prev_curve_fitarc=new_curve_fitarc

				if next_point==len(curve)-1:
					break
		
		print(breakpoints)


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

	return np.array(results_max_cartesian_error),np.array(results_max_cartesian_error_index),np.array(results_avg_cartesian_error),np.array(results_max_orientation_error), np.array(results_max_dz_error),np.array(results_avg_dz_error)




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

	fit_under_error(curve,curve_js,1.)

if __name__ == "__main__":
	main()
