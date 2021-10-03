import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from pandas import *
from toolbox_circular_fit import *


#####################3d circular fitting under error threshold, with equally divided breakpoint segments###############################



def fit_under_error(curve,max_error_threshold,d=50):

	###initialize
	breakpoints=[0,int(np.floor((len(curve)-1))/3),int(2*np.floor((len(curve)-1))/3),len(curve)-1]
	num_breakpoints=4
	max_error=999

	results_max_cartesian_error=[]
	results_max_cartesian_error_index=[]
	results_avg_cartesian_error=[]
	results_max_orientation_error=[]
	results_max_dz_error=[]
	results_avg_dz_error=[]


	while max_error>max_error_threshold:

		fit=stepwise_3dfitting(curve,breakpoints)

		###new breakpoints if max error out of threshold
		breakpoints=[0]
		num_breakpoints+=1
		for i in range(num_breakpoints-1):
			breakpoints.append(int(np.floor((i+1)*(len(curve)-1))/(num_breakpoints-1)))


		##############################check error (against fitting forward projected curve)##############################
		error=[]
		for i in range(len(fit)):
		    error_temp=np.linalg.norm(curve-fit[i],axis=1)
		    idx=np.argmin(error_temp)
		    error.append(error_temp[idx])

		error=np.array(error)
		max_error=np.max(error)
		print('max error: ', max_error)


		# max_error,max_cartesian_error_index,avg_cartesian_error,max_orientation_error=complete_points_check(curve_final_projection,curve,curve_R_pred,curve_R)
		# results_max_cartesian_error.append(max_error)
		# results_max_cartesian_error_index.append(max_cartesian_error_index)
		# results_avg_cartesian_error.append(avg_cartesian_error)
		# results_max_orientation_error.append(max_orientation_error)
		# results_max_dz_error.append(dz_error.max()-d)
		# results_avg_dz_error.append(dz_error.mean()-d)

	###plt
	###3D plot
	plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'gray')
	for i in range(len(breakpoints)-1):
		ax.scatter3D(fit[breakpoints[i]:breakpoints[i+1],0], fit[breakpoints[i]:breakpoints[i+1],1], fit[breakpoints[i]:breakpoints[i+1],2], c=fit[breakpoints[i]:breakpoints[i+1],2], cmap='Greens')
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
	# fit_under_error(curve,1)

	stepwise_3dfitting(curve,[    0,  8872,  9156, 12350, 13930, 14519, 17580])

if __name__ == "__main__":
	main()
