import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import *
from pwlfmd import *
import sys
import numpy as np
sys.path.append('toolbox')
from error_check import *
sys.path.append('data')
from cartesian2joint import direction2R
from pyquaternion import Quaternion


def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
 
	ndotu = planeNormal.dot(rayDirection)
	if abs(ndotu) < epsilon:
		raise RuntimeError("no intersection or line is within plane")
 
	w = rayPoint - planePoint
	si = -planeNormal.dot(w) / ndotu
	Psi = w + si * rayDirection + planePoint
	return Psi


def main():
	###All in base frame
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

	#########################fitting####################################
	###fitting back projection curve
	my_pwlf=MDFit(np.arange(len(curve_backproj)),curve_backproj)

	###slope calc breakpoints
	my_pwlf.break_slope_simplified(-1)
	print('num breakpoints: ',len(my_pwlf.break_points))

	my_pwlf.fit_with_breaks(my_pwlf.x_data[my_pwlf.break_points])

	###predict for the determined points
	xHat = np.arange(len(curve_backproj))
	curve_cartesian_pred = my_pwlf.predict_arb(xHat)

	####################################interpolate orientation##############################
	curve_R_pred=[]	
	curve_final_projection=np.zeros(np.shape(curve_cartesian_pred))
	for i in range(len(my_pwlf.break_points)-1):
		start=my_pwlf.break_points[i]
		if my_pwlf.break_points[i+1]==-1:
			end=len(curve)-1
		else:
			end=my_pwlf.break_points[i+1]-1

		if end==start:
			#no interpolation needed
			curve_R_pred.append(curve_R[start])
			curve_final_projection[start]=LinePlaneCollision(planeNormal=curve_direction[start], planePoint=curve[start], rayDirection=curve_R_pred[-1][:,-1], rayPoint=curve_cartesian_pred[start])
			continue


		q_start=Quaternion(matrix=curve_R[start])
		q_end=Quaternion(matrix=curve_R[end])
		for j in range(start,end+1):

			q_temp=Quaternion.slerp(q_start, q_end, float(j-start)/float(end-start)) 
			curve_R_pred.append(q_temp.rotation_matrix)
			curve_final_projection[j]=LinePlaneCollision(planeNormal=curve_direction[j], planePoint=curve[j], rayDirection=curve_R_pred[-1][:,-1], rayPoint=curve_cartesian_pred[j])



	####################################error checking####################################################
	print('calcualting final error')
	max_cartesian_error,max_orientation_error=complete_points_check(curve_final_projection,curve,curve_R_pred,curve_R)
	print('maximum cartesian error: ',max_cartesian_error)
	print('max orientation error: ', max_orientation_error)
	print('average error: ',calc_avg_error(curve_final_projection,curve))
	
if __name__ == "__main__":
	main()