import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from pandas import *
from toolbox_circular_fit import *

#####################3d circular fitting under error threshold, with stepwise incremental breakpoints###############################




def fit_under_error(curve,max_error_threshold,d=50):

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

	while breakpoints[-1]!=len(curve)-1:
		next_point= min(3000,len(curve)-1-breakpoints[-1])
		increment=100
		###start 1-seg fitting until reaching threshold
		###first test fitting
		prev_curve_fitarc,max_error=seg_3dfit(curve[breakpoints[-1]:breakpoints[-1]+next_point],p=[] if breakpoints[-1]>=0 else fit[-1][-1])


		if breakpoints[-1]+next_point==len(curve)-1 and max_error<=max_error_threshold:
			breakpoints.append(len(curve)-1)
			fit.append(prev_curve_fitarc)
			break

		###bp going backward to meet threshold
		if max_error>max_error_threshold:
			while True:
				next_point= min(next_point - increment,len(curve)-1-breakpoints[-1])
				curve_fitarc,max_error=seg_3dfit(curve[breakpoints[-1]:breakpoints[-1]+next_point],p=[] if breakpoints[-1]>=0 else fit[-1][-1])
				if max_error<=max_error_threshold:
					breakpoints.append(breakpoints[-1]+next_point)
					fit.append(curve_fitarc)
					break
		###bp going forward to get close to threshold
		else:
			while True:
				next_point= min(next_point + increment,len(curve)-1-breakpoints[-1])
				new_curve_fitarc, max_error=seg_3dfit(curve[breakpoints[-1]:breakpoints[-1]+next_point],p=[] if breakpoints[-1]>=0 else fit[-1][-1])
				if max_error>=max_error_threshold:
					breakpoints.append(breakpoints[-1]+next_point-increment)
					fit.append(prev_curve_fitarc)
					break
				prev_curve_fitarc=new_curve_fitarc

				if next_point==len(curve)-1:
					break
		
		print(breakpoints)


	##############################check error (against fitting forward projected curve)##############################
	fit_all=np.array(fit).reshape(-1,3)
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
	fit_under_error(curve,1.)

if __name__ == "__main__":
	main()
