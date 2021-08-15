import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import *
from pwlfmd import *
import sys
import numpy as np
sys.path.append('toolbox')
from error_check import *
from projection import LinePlaneCollision
sys.path.append('data')
from cartesian2joint import direction2R
from pyquaternion import Quaternion


def fit_test(curve,curve_backproj,curve_R,thresholds):
	results_max_cartesian_error_index=[]
	results_max_cartesian_error=[]
	results_max_orientation_error=[]
	results_avg_cartesian_error=[]
	results_num_breakpoints=[]
	for threshold in thresholds:
		#########################fitting####################################
		###fitting back projection curve
		my_pwlf=MDFit(np.arange(len(curve_backproj)),curve_backproj)

		###slope calc breakpoints
		my_pwlf.break_slope_simplified(threshold)
		results_num_breakpoints.append(len(my_pwlf.break_points))

		my_pwlf.fit_with_breaks(my_pwlf.x_data[my_pwlf.break_points])

		###predict for the determined points
		xHat = np.arange(len(curve_backproj))
		curve_cartesian_pred = my_pwlf.predict_arb(xHat)

		####################################interpolate orientation##############################
		curve_R_pred=[]	
		curve_final_projection=np.zeros(np.shape(curve_cartesian_pred))
		for i in range(len(my_pwlf.break_points)-1):

			###determine start and end interpolation index
			start=my_pwlf.break_points[i]
			if my_pwlf.break_points[i+1]==-1:
				end=len(curve)-1
			else:
				end=my_pwlf.break_points[i+1]-1

			if end==start:
				#no interpolation needed
				curve_R_pred.append(curve_R[start])
				curve_final_projection[start]=LinePlaneCollision(planeNormal=curve_R[start][:,-1], planePoint=curve[start], rayDirection=curve_R_pred[-1][:,-1], rayPoint=curve_cartesian_pred[start])
				continue

			###quaternion spherical linear interpolation
			q_start=Quaternion(matrix=curve_R[start])
			q_end=Quaternion(matrix=curve_R[end])
			for j in range(start,end+1):
				###calculate orientation
				q_temp=Quaternion.slerp(q_start, q_end, float(j-start)/float(end-start)) 
				curve_R_pred.append(q_temp.rotation_matrix)
				###get line/surface intersection point
				curve_final_projection[j]=LinePlaneCollision(planeNormal=curve_R[j][:,-1], planePoint=curve[j], rayDirection=curve_R_pred[-1][:,-1], rayPoint=curve_cartesian_pred[j])
		###calculating error
		max_cartesian_error,max_cartesian_error_index,avg_cartesian_error,max_orientation_error=complete_points_check(curve_final_projection,curve,curve_R_pred,curve_R)
		###attach to results
		results_max_cartesian_error.append(max_cartesian_error)
		results_max_cartesian_error_index.append(max_cartesian_error_index)
		results_avg_cartesian_error.append(avg_cartesian_error)
		results_max_orientation_error.append(max_orientation_error)

	# plt.figure()
	# ax = plt.axes(projection='3d')
	# ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'gray')
	# ax.scatter3D(curve_final_projection[:,0], curve_final_projection[:,1], curve_final_projection[:,2], c=curve_final_projection[:,2], cmap='Greens')
	# ax.scatter3D(curve_cartesian_pred[:,0], curve_cartesian_pred[:,1], curve_cartesian_pred[:,2], c=curve_cartesian_pred[:,2], cmap='Blues')
	# plt.show()

	return results_num_breakpoints,results_max_cartesian_error,results_max_cartesian_error_index,results_avg_cartesian_error,results_max_orientation_error
		
def main():
	###All in base frame
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("data/from_interp/Curve_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T
	
	###back projection
	d=50			###offset
	curve_backproj=curve-d*curve_direction

	#get orientation
	curve_R=[]
	for i in range(len(curve)):
		try:
			R_curve=direction2R(curve_direction[i],-curve[i+1]+curve[i])
		except:
			traceback.print_exc()
			pass
		curve_R.append(R_curve)

	#########################fitting tests####################################
	thresholds=[5.00E-04,1.00E-04,1.00E-05,7.81E-06,3.91E-06,3.20E-06,1.95E-06,1.00E-06,8.00E-07,5.00E-07,4.00E-07]
	# thresholds=[5.00E-07]
	results_num_breakpoints,results_max_cartesian_error,results_max_cartesian_error_index,results_avg_cartesian_error,results_max_orientation_error=\
		fit_test(curve,curve_backproj,curve_R,thresholds)

	###output to csv
	df=DataFrame({'num_breakpoints':results_num_breakpoints,'max_cartesian_error':results_max_cartesian_error,'max_cartesian_error_index':results_max_cartesian_error_index,'avg_cartesian_error':results_avg_cartesian_error,'max_orientation_error':results_max_orientation_error})
	df.to_csv('results/from_interp/cartesian_fit_results.csv',header=True,index=False)


	plt.figure()
	plt.plot(results_num_breakpoints,results_max_cartesian_error)
	plt.show()
if __name__ == "__main__":
	main()