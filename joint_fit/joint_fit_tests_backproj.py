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
import traceback

def fit_test(curve,curve_js,thresholds,d=50):
	results_max_cartesian_error_index=[]
	results_max_cartesian_error=[]
	results_max_orientation_error=[]
	results_avg_cartesian_error=[]
	results_num_breakpoints=[]
	results_max_dz_error=[]
	results_avg_dz_error=[]
	for threshold in thresholds:
		#########################fitting####################################
		###fitting back projection curve in joint space
		my_pwlf=MDFit(np.arange(len(curve_js)),curve_js)

		###slope calc breakpoints
		my_pwlf.break_slope_simplified(threshold)
		results_num_breakpoints.append(len(my_pwlf.break_points))

		my_pwlf.fit_with_breaks(my_pwlf.lam_data[my_pwlf.break_points])

		###predict at every data index
		xHat = np.arange(0,len(curve_js))
		curve_js_pred = my_pwlf.predict_arb(xHat)

		####################################get cartesian and orientation##############################
		curve_cartesian_pred=[]
		curve_R_pred=[]
		curve_R=[]
		curve_final_projection=[]
		curve_js_cartesian=[]
		dz_error=[]
		for i in range(len(curve_js_pred)):
			fwdkin_result=fwd(curve_js_pred[i])
			curve_cartesian_pred.append(fwdkin_result.p)
			curve_R_pred.append(fwdkin_result.R)
			try:
				fwdkin_result2=fwd(curve_js[i])
				curve_R.append(fwdkin_result2.R)
				curve_js_cartesian.append(fwdkin_result2.p)


				###project forward onto curve surface, all in reference frame
				intersection=LinePlaneCollision(planeNormal=curve_R[i][:,-1], planePoint=curve[i], rayDirection=curve_R_pred[i][:,-1], rayPoint=curve_cartesian_pred[i])
				d_z=np.linalg.norm(intersection-curve_cartesian_pred[i])
				dz_error.append(d_z)
				curve_final_projection.append(intersection)
			except:
				# traceback.print_exc()
				pass

		dz_error=np.array(dz_error)
		###calculating error
		max_cartesian_error,max_cartesian_error_index,avg_cartesian_error,max_orientation_error=complete_points_check(curve_final_projection,curve,curve_R_pred,curve_R)
		###attach to results
		results_max_cartesian_error.append(max_cartesian_error)
		results_max_cartesian_error_index.append(max_cartesian_error_index)
		results_avg_cartesian_error.append(avg_cartesian_error)
		results_max_orientation_error.append(max_orientation_error)
		results_max_dz_error.append(dz_error.max()-d)
		results_avg_dz_error.append(dz_error.mean()-d)

	curve_final_projection=np.array(curve_final_projection)
	curve_cartesian_pred=np.array(curve_cartesian_pred)
	curve_js_cartesian=np.array(curve_js_cartesian)
	# plt.figure()
	# plt.scatter(np.arange(len(curve_cartesian_pred)),curve_cartesian_pred[:,0],s=1)
	# plt.scatter(np.arange(len(curve_cartesian_pred)),curve_cartesian_pred[:,1],s=1)
	# plt.scatter(np.arange(len(curve_cartesian_pred)),curve_cartesian_pred[:,2],s=1)
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
	# plt.show()

	return np.array(results_num_breakpoints),np.array(results_max_cartesian_error),np.array(results_max_cartesian_error_index),np.array(results_avg_cartesian_error),np.array(results_max_orientation_error), np.array(results_max_dz_error),np.array(results_avg_dz_error)
		

def main():
	###read actual curve
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

	
	###read interpolated curves in joint space
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("data/from_interp/Curve_backproj_js.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	#########################fitting tests####################################
	# thresholds=[0.000390625,9.77E-05,5.00E-05,4.00E-05,3.00E-05]	#for cad points
	thresholds=[1,0.5,1.00E-01,1.00E-02,1.00E-03,1.00E-04]#,1.00E-07]	#for interp points

	results_num_breakpoints,results_max_cartesian_error,results_max_cartesian_error_index,results_avg_cartesian_error,results_max_orientation_error, results_max_dz_error, results_avg_dz_error=\
		fit_test(curve,curve_js,thresholds,d=50)

	###output to csv
	df=DataFrame({'num_breakpoints':results_num_breakpoints,'max_cartesian_error (mm)':results_max_cartesian_error,'max_cartesian_error_index (mm)':results_max_cartesian_error_index,'avg_cartesian_error (mm)':results_avg_cartesian_error,'max_orientation_error  (rad)':results_max_orientation_error,'max_z_error (mm)':results_max_dz_error,'average_z_error (mm)':results_avg_dz_error})
	df.to_csv('results/from_interp/joint_fit_results.csv',header=True,index=False)

	plt.figure()
	plt.plot(results_num_breakpoints,results_max_cartesian_error)
	plt.show()

if __name__ == "__main__":
	main()