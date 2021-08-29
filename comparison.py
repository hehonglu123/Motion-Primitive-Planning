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
from general_robotics_toolbox import *

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

	num_points=500		### try fitting every 100 points


	results_max_cartesian_error_index_joint=[]
	results_max_cartesian_error_joint=[]
	results_max_orientation_error_joint=[]
	results_avg_cartesian_error_joint=[]
	results_max_dz_error_joint=[]
	results_avg_dz_error_joint=[]
	results_max_cartesian_error_index_cartesian=[]
	results_max_cartesian_error_cartesian=[]
	results_max_orientation_error_cartesian=[]
	results_avg_cartesian_error_cartesian=[]
	results_max_dz_error_cartesian=[]
	results_avg_dz_error_cartesian=[]

	breakpoints=[0, 427, 853, 1278, 1714, 2166, 2643, 3154, 3713, 4344, 5095, 5985, 7404, 8367, 9128, 9780, 10360, 10890, 11383, 11850, 12300, 12740, 13180, 13581, 13990, 14415, 14860, 15332, 15835, 16376, 16944, 17579]

	# for i in range(0,len(curve),num_points):
	# end=min(i+num_points,len(curve))
	for r in range(len(breakpoints)-1):
		i=breakpoints[r]
		end=breakpoints[r+1]
		act_num_points=len(curve[i:end])
		x_data=np.arange(i,end)
		###########################cartesian fit########################
		
		my_pwlf=MDFit(x_data,curve_backproj[i:end])
		my_pwlf.fit_with_breaks([i,end-1])

		###predict for the determined points
		curve_cartesian_pred = my_pwlf.predict_arb(x_data)

		curve_R_pred=[]	
		dz_error=[]
		curve_final_projection=[]
		###axis-angle interpolation
		R_temp=np.dot(curve_R[i].T,curve_R[end-1])
		k,theta=R2rot(R_temp)
		for j in range(i,end):
			
			###calculate orientation
			theta_temp=theta*float(j-i)/float(end-1-i)
			R_interp=q2R(rot2q(k,theta_temp))
			curve_R_pred.append(np.dot(curve_R[i],R_interp))
			###get line/surface intersection point
			intersection=LinePlaneCollision(planeNormal=curve_R[j][:,-1], planePoint=curve[j], rayDirection=curve_R_pred[-1][:,-1], rayPoint=curve_cartesian_pred[j-i])
			curve_final_projection.append(intersection)
			d_z=np.linalg.norm(intersection-curve_cartesian_pred[j-i])
			dz_error.append(d_z)

		dz_error=np.array(dz_error)
		###calculating error
		max_cartesian_error,max_cartesian_error_index,avg_cartesian_error,max_orientation_error=complete_points_check(curve_final_projection,curve[i:end],curve_R_pred,curve_R)
		results_max_cartesian_error_cartesian.append(max_cartesian_error)
		results_max_cartesian_error_index_cartesian.append(max_cartesian_error_index)
		results_avg_cartesian_error_cartesian.append(avg_cartesian_error)
		results_max_orientation_error_cartesian.append(max_orientation_error)
		results_max_dz_error_cartesian.append(dz_error.max()-d)
		results_avg_dz_error_cartesian.append(dz_error.mean()-d)


		###########################joint fit###############################
		my_pwlf=MDFit(x_data,curve_js[i:end])
		my_pwlf.fit_with_breaks([i,end-1])

		###predict at every data index
		curve_js_pred = my_pwlf.predict_arb(x_data)

		curve_cartesian_pred=[]
		curve_R_pred=[]
		curve_final_projection=[]
		curve_js_cartesian=[]
		dz_error=[]
		for j in range(i,end):
			fwdkin_result=fwd(curve_js_pred[j-i])
			curve_cartesian_pred.append(1000.*fwdkin_result.p)
			curve_R_pred.append(fwdkin_result.R)
			try:
				fwdkin_result2=fwd(curve_js[i])
				curve_js_cartesian.append(1000.*fwdkin_result2.p)


				###project forward onto curve surface, all in reference frame
				intersection=LinePlaneCollision(planeNormal=curve_R[j][:,-1], planePoint=curve[j], rayDirection=curve_R_pred[j-i][:,-1], rayPoint=curve_cartesian_pred[j-i])
				d_z=np.linalg.norm(intersection-curve_cartesian_pred[j-i])
				dz_error.append(d_z)
				curve_final_projection.append(intersection)
			except:
				# traceback.print_exc()
				pass

		dz_error=np.array(dz_error)
		###calculating error
		max_cartesian_error,max_cartesian_error_index,avg_cartesian_error,max_orientation_error=complete_points_check(curve_final_projection,curve[i:end],curve_R_pred,curve_R[i:end])
		results_max_cartesian_error_joint.append(max_cartesian_error)
		results_max_cartesian_error_index_joint.append(max_cartesian_error_index)
		results_avg_cartesian_error_joint.append(avg_cartesian_error)
		results_max_orientation_error_joint.append(max_orientation_error)
		results_max_dz_error_joint.append(dz_error.max()-d)
		results_avg_dz_error_joint.append(dz_error.mean()-d)


	###output to csv
	df=DataFrame({'max_cartesian_error_joint (mm)':results_max_cartesian_error_joint,'max_cartesian_error_index_joint (mm)':results_max_cartesian_error_index_joint,'avg_cartesian_error_joint (mm)':results_avg_cartesian_error_joint,'max_orientation_error_joint  (rad)':results_max_orientation_error_joint,'max_z_error_joint (mm)':results_max_dz_error_joint,'average_z_error_joint (mm)':results_avg_dz_error_joint,\
		'max_cartesian_error_cartesian (mm)':results_max_cartesian_error_cartesian,'max_cartesian_error_index_cartesian (mm)':results_max_cartesian_error_index_cartesian,'avg_cartesian_error_cartesian (mm)':results_avg_cartesian_error_cartesian,'max_orientation_error _cartesian (rad)':results_max_orientation_error_cartesian,'max_z_error_cartesian (mm)':results_max_dz_error_cartesian,'average_z_error_cartesian (mm)':results_avg_dz_error_cartesian})
	df.to_csv('results/from_interp/comparison.csv',header=True,index=False)


if __name__ == "__main__":
	main()