import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import *
from pwlfmd import *
import numpy as np
import sys
sys.path.append('toolbox')
from error_check import *
from robot_def import *
from projection import LinePlaneCollision


def fit_test(curve,curve_js,thresholds):
	results_max_cartesian_error_index=[]
	results_max_cartesian_error=[]
	results_max_orientation_error=[]
	results_avg_cartesian_error=[]
	results_num_breakpoints=[]
	for threshold in thresholds:
		#########################fitting####################################
		###fitting back projection curve in joint space
		my_pwlf=MDFit(np.arange(len(curve_js)),curve_js)

		###slope calc breakpoints
		my_pwlf.break_slope_simplified(threshold)
		results_num_breakpoints.append(len(my_pwlf.break_points))

		my_pwlf.fit_with_breaks(my_pwlf.x_data[my_pwlf.break_points])

		###predict at every data index
		xHat = np.arange(len(curve_js))
		curve_js_pred = my_pwlf.predict_arb(xHat)

		####################################get cartesian and orientation##############################
		curve_cartesian_pred=[]
		curve_R_pred=[]
		curve_R=[]
		curve_final_projection=[]
		curve_js_cartesian=[]
		for i in range(len(curve_js_pred)):
			fwdkin_result=fwd(curve_js_pred[i])
			curve_cartesian_pred.append(1000.*fwdkin_result.p)
			curve_R_pred.append(fwdkin_result.R)

			fwdkin_result2=fwd(curve_js[i])
			curve_R.append(fwdkin_result2.R)
			curve_js_cartesian.append(1000.*fwdkin_result2.p)

			###project forward onto curve surface, all in reference frame
			curve_final_projection.append(LinePlaneCollision(planeNormal=curve_R[i][:,-1], planePoint=curve[i], rayDirection=curve_R_pred[i][:,-1], rayPoint=curve_cartesian_pred[i]))
			

		###calculating error
		max_cartesian_error,max_cartesian_error_index,avg_cartesian_error,max_orientation_error=complete_points_check(curve_final_projection,curve,curve_R_pred,curve_R)
		###attach to results
		results_max_cartesian_error.append(max_cartesian_error)
		results_max_cartesian_error_index.append(max_cartesian_error_index)
		results_avg_cartesian_error.append(avg_cartesian_error)
		results_max_orientation_error.append(max_orientation_error)

	curve_final_projection=np.array(curve_final_projection)
	curve_cartesian_pred=np.array(curve_cartesian_pred)
	# curve_js_cartesian=np.array(curve_js_cartesian)
	plt.figure()
	plt.scatter(np.arange(len(curve_cartesian_pred)),curve_cartesian_pred[:,0],s=1)
	plt.scatter(np.arange(len(curve_cartesian_pred)),curve_cartesian_pred[:,1],s=1)
	plt.scatter(np.arange(len(curve_cartesian_pred)),curve_cartesian_pred[:,2],s=1)
	# plt.scatter(np.arange(len(curve_cartesian_pred)),curve_js[:,0],s=1)
	# plt.scatter(np.arange(len(curve_cartesian_pred)),curve_js[:,1],s=1)
	# plt.scatter(np.arange(len(curve_cartesian_pred)),curve_js[:,2],s=1)
	# plt.scatter(np.arange(len(curve_cartesian_pred)),curve_js[:,3],s=1)
	# plt.scatter(np.arange(len(curve_cartesian_pred)),curve_js[:,4],s=1)
	# plt.scatter(np.arange(len(curve_cartesian_pred)),curve_js[:,5],s=1)
	# ax = plt.axes(projection='3d')
	# ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'gray')
	# ax.scatter3D(curve_final_projection[:,0], curve_final_projection[:,1], curve_final_projection[:,2], c=curve_final_projection[:,2], cmap='Greens')
	# ax.scatter3D(curve_cartesian_pred[:,0], curve_cartesian_pred[:,1], curve_cartesian_pred[:,2], c=curve_cartesian_pred[:,2], cmap='Blues')
	# ax.scatter3D(curve_js_cartesian[:,0], curve_js_cartesian[:,1], curve_js_cartesian[:,2], c=curve_js_cartesian[:,2], cmap='Blues')
	plt.show()

	return results_num_breakpoints,results_max_cartesian_error,results_max_cartesian_error_index,results_avg_cartesian_error,results_max_orientation_error
		

def main():
	###read actual curve
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("data/Curve_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

	
	###read interpolated curves in joint space
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("data/Curve_backproj_js.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	#########################fitting tests####################################
	# thresholds=[0.000390625,9.77E-05,5.00E-05,4.00E-05,3.00E-05]
	thresholds=[0.000390625]
	results_num_breakpoints,results_max_cartesian_error,results_max_cartesian_error_index,results_avg_cartesian_error,results_max_orientation_error=\
		fit_test(curve,curve_js,thresholds)

	print(results_max_cartesian_error)

	plt.figure()
	plt.plot(results_num_breakpoints,results_max_cartesian_error)
	plt.show()

if __name__ == "__main__":
	main()