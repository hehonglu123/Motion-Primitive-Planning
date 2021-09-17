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

def fit_under_error(curve,curve_js,max_error_threshold,d=0):

	###initialize
	lam_data=np.arange(len(curve_js))
	my_pwlf=MDFit(lam_data,curve_js)
	breakpoints=[0,len(lam_data)-1]
	max_error=999

	results_max_cartesian_error_index=[]
	results_max_cartesian_error=[]
	results_max_orientation_error=[]
	results_avg_cartesian_error=[]
	results_num_breakpoints=[]
	results_max_dz_error=[]
	results_avg_dz_error=[]

	while max_error>max_error_threshold:

		if (max_error!=999):

			breakpoints.append(next_breakpoint)
			breakpoints.sort()
		my_pwlf.fit_with_breaks(breakpoints)


		###fit
		curve_js_pred=my_pwlf.predict()

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
				d_z=0
				dz_error.append(d_z)
			except:
				traceback.print_exc()
				pass

		dz_error=np.array(dz_error)
		###calculating error
		max_error,max_cartesian_error_index,avg_cartesian_error,max_orientation_error=complete_points_check(curve_cartesian_pred,curve,curve_R_pred,curve_R)
		###attach to results
		results_max_cartesian_error.append(max_error)
		results_max_cartesian_error_index.append(max_cartesian_error_index)
		results_avg_cartesian_error.append(avg_cartesian_error)
		results_max_orientation_error.append(max_orientation_error)
		results_max_dz_error.append(dz_error.max()-d)
		results_avg_dz_error.append(dz_error.mean()-d)

		##################find next breakpoint##################
		temp_fit=[]
		for i in range(len(breakpoints)-1):
			if i!=len(breakpoints)-2:
				temp_lam=np.arange(breakpoints[i],breakpoints[i+1])
			else:
				temp_lam=np.arange(breakpoints[i],breakpoints[i+1]+1)
			interp=interp1d(np.array([breakpoints[i],breakpoints[i+1]]),np.array([curve_js[breakpoints[i]],curve_js[breakpoints[i+1]]]),axis=0)
			temp_fit.append(interp(temp_lam))

		temp_fit=np.concatenate( temp_fit, axis=0 ).reshape(len(curve_js),len(curve_js[0]))
		errors=np.linalg.norm(temp_fit-curve_js,axis=1)
		next_breakpoint=np.argsort(errors)[-1]
		print(max_error,len(breakpoints))

	return np.array(results_max_cartesian_error),np.array(results_max_cartesian_error_index),np.array(results_avg_cartesian_error),np.array(results_max_orientation_error), np.array(results_max_dz_error),np.array(results_avg_dz_error)


def main():
	###read actual curve
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

	
	###read interpolated curves in joint space
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("../data/from_interp/Curve_js.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	#########################fitting tests####################################
	
	results_max_cartesian_error,results_max_cartesian_error_index,results_avg_cartesian_error,results_max_orientation_error, results_max_dz_error, results_avg_dz_error=\
		fit_under_error(curve,curve_js,1)

	###output to csv
	df=DataFrame({'max_cartesian_error (mm)':results_max_cartesian_error,'max_cartesian_error_index (mm)':results_max_cartesian_error_index,'avg_cartesian_error (mm)':results_avg_cartesian_error,'max_orientation_error  (rad)':results_max_orientation_error,'max_z_error (mm)':results_max_dz_error,'average_z_error (mm)':results_avg_dz_error})
	df.to_csv('../results/from_interp/joint_fit_results.csv',header=True,index=False)

	plt.figure()
	plt.plot(results_max_cartesian_error)
	plt.show()








	
if __name__ == "__main__":
	main()