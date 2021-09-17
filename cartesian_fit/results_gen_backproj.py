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
sys.path.append('../data')
from cartesian2joint import direction2R

def fit_under_error(curve,curve_backproj,curve_R,max_error_threshold,d=50):

	###initialize
	lam_data=np.arange(len(curve_backproj))
	my_pwlf=MDFit(lam_data,curve_backproj)
	breakpoints=[0,len(lam_data)-1]
	max_error=999

	results_max_cartesian_error=[]
	results_max_cartesian_error_index=[]
	results_avg_cartesian_error=[]
	results_max_orientation_error=[]
	results_max_dz_error=[]
	results_avg_dz_error=[]


	while max_error>max_error_threshold:

		if (max_error!=999):

			breakpoints.append(next_breakpoint)
			breakpoints.sort()
		my_pwlf.fit_with_breaks(breakpoints)

		
		curve_cartesian_pred=my_pwlf.predict()
		
		curve_R_pred=[]	
		dz_error=[]
		curve_final_projection=np.zeros(np.shape(curve_cartesian_pred))

		
		temp_fit=[]
		for i in range(len(breakpoints)-1):
			##############################check error (against fitting forward projected curve)##############################
			###determine start and end interpolation index
			start=breakpoints[i]
			end=breakpoints[i+1]
			

			###axis-angle interpolation
			R_temp=np.dot(curve_R[start].T,curve_R[end])
			k,theta=R2rot(R_temp)


			if end==breakpoints[-1]:
				end+=1

			for j in range(start,end):
				###calculate orientation
				theta_temp=theta*float(j-start)/float(end-start)
				R_interp=q2R(rot2q(k,theta_temp))
				curve_R_pred.append(np.dot(curve_R[start],R_interp))
				###get line/surface intersection point
				intersection=LinePlaneCollision(planeNormal=curve_R[j][:,-1], planePoint=curve[j], rayDirection=curve_R_pred[-1][:,-1], rayPoint=curve_cartesian_pred[j])
				curve_final_projection[j]=intersection
				d_z=np.linalg.norm(intersection-curve_cartesian_pred[j])
				dz_error.append(d_z)



			#########################find next breakpoint#########################
			if i!=len(breakpoints)-2:
				temp_lam=np.arange(breakpoints[i],breakpoints[i+1])
			else:
				temp_lam=np.arange(breakpoints[i],breakpoints[i+1]+1)
			interp=interp1d(np.array([breakpoints[i],breakpoints[i+1]]),np.array([curve_backproj[breakpoints[i]],curve_backproj[breakpoints[i+1]]]),axis=0)
			temp_fit.append(interp(temp_lam))

		temp_fit=np.concatenate( temp_fit, axis=0 ).reshape(len(curve_backproj),len(curve_backproj[0]))
		errors=np.linalg.norm(temp_fit-curve_backproj,axis=1)
		next_breakpoint=np.argsort(errors)[-1]

		##############################check error (against fitting forward projected curve)##############################
		dz_error=np.clip(np.array(dz_error)-d,0,999)		###dz can't be smaller than 0
		max_error,max_cartesian_error_index,avg_cartesian_error,max_orientation_error=complete_points_check(curve_final_projection,curve,curve_R_pred,curve_R)
		results_max_cartesian_error.append(max_error)
		results_max_cartesian_error_index.append(max_cartesian_error_index)
		results_avg_cartesian_error.append(avg_cartesian_error)
		results_max_orientation_error.append(max_orientation_error)
		results_max_dz_error.append(dz_error.max())
		results_avg_dz_error.append(dz_error.mean())

		print(max_error,len(breakpoints))

	return np.array(results_max_cartesian_error),np.array(results_max_cartesian_error_index),np.array(results_avg_cartesian_error),np.array(results_max_orientation_error), np.array(results_max_dz_error),np.array(results_avg_dz_error)

def main():
	###All in base frame
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../data/from_interp/Curve_in_base_frame.csv", names=col_names)
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
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../data/from_interp/Curve_backproj_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()
	curve_backproj=np.vstack((curve_x, curve_y, curve_z)).T
	curve_backproj_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

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
	results_max_cartesian_error,results_max_cartesian_error_index,results_avg_cartesian_error,results_max_orientation_error, results_max_dz_error, results_avg_dz_error=\
		fit_under_error(curve,curve_backproj,curve_R,1,d=d)

	###output to csv
	df=DataFrame({'max_cartesian_error (mm)':results_max_cartesian_error,'max_cartesian_error_index (mm)':results_max_cartesian_error_index,'avg_cartesian_error (mm)':results_avg_cartesian_error,'max_orientation_error  (rad)':results_max_orientation_error,'max_z_error (mm)':results_max_dz_error,'average_z_error (mm)':results_avg_dz_error})
	df.to_csv('../results/from_interp/cartesian_fit_results_backproj.csv',header=True,index=False)


	plt.figure()
	plt.plot(results_max_cartesian_error)
	plt.show()	
if __name__ == "__main__":
	main()