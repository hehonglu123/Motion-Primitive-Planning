import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from pandas import *
from toolbox_circular_fit import *






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

		fit=[]
		
		for i in range(len(breakpoints)-1):
			if len(breakpoints)==2:
				###first fit
				curve_fitarc,curve_fitcircle=circle_fit(curve)
				fit=curve_fitarc
			else:
				seg2fit=curve[breakpoints[i]:breakpoints[i+1]]
				if i==0:

					curve_fitarc,curve_fit_circle=circle_fit(seg2fit)
					fit.append(curve_fitarc)


					# plt.figure()
					# ax = plt.axes(projection='3d')
					# ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'gray')
					# ax.scatter3D(curve_fitarc[:,0], curve_fitarc[:,1], curve_fitarc[:,2], c=curve_fitarc[:,2], cmap='Greens')
					# plt.show()
				else:
					curve_fitarc,curve_fit_circle=circle_fit(seg2fit,p=fit[-1][-1])
					fit.append(curve_fitarc)


					# plt.figure()
					# ax = plt.axes(projection='3d')
					# ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'gray')
					# ax.scatter3D(curve_fitarc[:,0], curve_fitarc[:,1], curve_fitarc[:,2], c=curve_fitarc[:,2], cmap='Greens')
					# plt.show()

		###new breakpoints if max error out of threshold
		breakpoints=[0]
		num_breakpoints+=1
		for i in range(num_breakpoints-1):
			breakpoints.append(int(np.floor((i+1)*(len(curve)-1))/(num_breakpoints-1)))


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
	fit_under_error(curve,1)

if __name__ == "__main__":
	main()
