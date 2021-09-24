import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pandas import *
sys.path.append('../')
from pwlfmd import *
import numpy as np
from scipy.interpolate import interp1d
sys.path.append('../toolbox')
from projection import LinePlaneCollision
from error_check import *
from robot_def import *
sys.path.append('../data')
from cartesian2joint import direction2R

def fit_under_error(curve,curve_backproj_js,max_error_threshold,d=50):

	###initialize
	lam_data=np.arange(len(curve_backproj_js))
	my_pwlf=MDFit(lam_data,curve_backproj_js)
	breakpoints=[0,len(lam_data)-1]
	max_error=999

	results_max_cartesian_error=[]
	results_avg_cartesian_error=[]
	results_max_backproj_error=[]
	results_avg_backproj_error=[]
	results_max_total_error=[]
	results_avg_total_error=[]
	results_max_error_index=[]

	while max_error>max_error_threshold and len(breakpoints)<100:

		if (max_error!=999):

			breakpoints.append(next_breakpoint)
			breakpoints.sort()
		my_pwlf.fit_with_breaks(breakpoints)


		###fit
		curve_backproj_js_pred=my_pwlf.predict()

		####################################get cartesian and orientation##############################
		curve_cartesian_pred=[]
		curve_R_pred=[]
		curve_R=[]
		curve_backproj=[]
		curve_final_projection=[]
		dz_error=[]
		for i in range(len(curve_backproj_js_pred)):
			fwdkin_result=fwd(curve_backproj_js_pred[i])
			curve_cartesian_pred.append(fwdkin_result.p)
			curve_R_pred.append(fwdkin_result.R)
			try:
				fwdkin_result2=fwd(curve_backproj_js[i])
				curve_R.append(fwdkin_result2.R)
				curve_backproj.append(fwdkin_result2.p)

				###project forward onto curve surface, all in reference frame
				intersection=LinePlaneCollision(planeNormal=curve_R[i][:,-1], planePoint=curve[i], rayDirection=curve_R_pred[i][:,-1], rayPoint=curve_cartesian_pred[i])
				d_z=np.linalg.norm(intersection-curve_cartesian_pred[i])
				dz_error.append(d_z)
				curve_final_projection.append(intersection)
			except:
				traceback.print_exc()
				pass

		curve_backproj=np.array(curve_backproj)
		curve_final_projection=np.array(curve_final_projection)
		curve_cartesian_pred=np.array(curve_cartesian_pred)

		dz_error=np.clip(np.array(dz_error)-d,0,999)		###dz can't be smaller than 0
		###calculating error
		max_error,avg_cartesian_error,max_cartesian_error_backproj,avg_cartesian_error_backproj,max_total_error,avg_total_error,max_error_index \
		=complete_points_check2(curve_cartesian_pred,curve_backproj,curve_final_projection,curve)

		results_max_cartesian_error.append(max_error)
		results_avg_cartesian_error.append(avg_cartesian_error)
		results_max_backproj_error.append(max_cartesian_error_backproj)
		results_avg_backproj_error.append(avg_cartesian_error_backproj)
		results_max_total_error.append(max_total_error)
		results_avg_total_error.append(avg_total_error)
		results_max_error_index.append(max_error_index)


		###large error debug
		# if max_error>900:
		# 	curve_cartesian_pred=np.array(curve_cartesian_pred)
		# 	curve_final_projection=np.array(curve_final_projection)
		# 	plt.figure()
		# 	ax = plt.axes(projection='3d')
		# 	ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'gray')
		# 	ax.scatter3D(curve_final_projection[:,0], curve_final_projection[:,1], curve_final_projection[:,2], c=curve_final_projection[:,2], cmap='Greens')
		# 	ax.scatter3D(curve_cartesian_pred[:,0], curve_cartesian_pred[:,1], curve_cartesian_pred[:,2], c=curve_cartesian_pred[:,2], cmap='Blues')
		# 	plt.show()

		##################find next breakpoint##################
		temp_fit=[]
		for i in range(len(breakpoints)-1):
			if i!=len(breakpoints)-2:
				temp_lam=np.arange(breakpoints[i],breakpoints[i+1])
			else:
				temp_lam=np.arange(breakpoints[i],breakpoints[i+1]+1)
			interp=interp1d(np.array([breakpoints[i],breakpoints[i+1]]),np.array([curve_backproj_js[breakpoints[i]],curve_backproj_js[breakpoints[i+1]]]),axis=0)
			temp_fit.append(interp(temp_lam))

		temp_fit=np.concatenate( temp_fit, axis=0 ).reshape(len(curve_backproj_js),len(curve_backproj_js[0]))
		errors=np.linalg.norm(temp_fit-curve_backproj_js,axis=1)
		next_breakpoint=np.argsort(errors)[-1]
		print(max_error,len(breakpoints))

	return np.array(results_max_cartesian_error),np.array(results_avg_cartesian_error),np.array(results_max_backproj_error),np.array(results_avg_backproj_error), \
		np.array(results_max_total_error),np.array(results_avg_total_error),np.array(results_max_error_index)

def main():
	###read actual curve
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../data/from_cad/Curve_in_base_frame.csv", names=col_names)
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
	data = read_csv("../data/from_cad/Curve_backproj_js.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_backproj_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	#########################fitting tests####################################
	
	results_max_cartesian_error,results_avg_cartesian_error,results_max_backproj_error,results_avg_backproj_error,results_max_total_error,\
	results_avg_total_error,results_max_error_index=fit_under_error(curve,curve_backproj_js,1,d=50)

	###output to csv
	df=DataFrame({'max_cartesian_error':results_max_cartesian_error,'avg_cartesian_error':results_avg_cartesian_error,\
		'max_projection_error':results_max_backproj_error,'avg_projection_error':results_avg_backproj_error,\
		'max_total_error':results_max_total_error,'avg_total_error':results_avg_total_error,'max_error_idx':results_max_error_index})	
	df.to_csv('../results/from_cad/joint_fit_results_backproj.csv',header=True,index=False)

	plt.figure()
	plt.plot(results_max_cartesian_error)
	plt.show()








	
if __name__ == "__main__":
	main()