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
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("data/Curve_dense.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z))
	curve_direction_ref=curve_direction.T
	
	###back projection
	d=50			###offset
	curve_backproj=curve-d*curve_direction_ref

	###reference frame transformation
	R=np.array([[0,0,1.],
				[1.,0,0],
				[0,1.,0]])
	T=np.array([[2700.],[-800.],[500.]])
	H=np.vstack((np.hstack((R,T)),np.array([0,0,0,1])))
	#convert curve direction to base frame
	curve_direction_base=np.dot(R,curve_direction).T
	curve_R=[]
	curve_base=np.zeros(curve.shape)
	for i in range(len(curve)):
		curve_base[i]=np.dot(H,np.hstack((curve_backproj[i],[1])).T)[:-1]
		try:
			R_curve=direction2R(curve_direction_base[i],-curve_base[i]+curve_base[i-1])
		except:
			traceback.print_exc()
			pass
		curve_R.append(np.dot(R.T,R_curve))

	###insert initial orientation
	curve_R.insert(0,curve_R[0])

	#########################fitting####################################
	###fitting back projection curve
	my_pwlf=MDFit(np.arange(len(curve_backproj)),curve_backproj)

	###slope calc breakpoints
	my_pwlf.break_slope_simplified(0.00001)
	print('num breakpoints: ',len(my_pwlf.break_points))

	my_pwlf.fit_with_breaks(my_pwlf.x_data[my_pwlf.break_points])

	###predict for the determined points
	xHat = np.arange(len(curve_backproj))
	curve_cartesian_pred = my_pwlf.predict_arb(xHat)

	####################################interpolate orientation##############################
	curve_R_pred=[]	
	curve_final_projection=np.zeros(np.shape(curve_cartesian_pred))
	for i in range(len(my_pwlf.break_points)-1):
		q_start=Quaternion(matrix=curve_R[my_pwlf.break_points[i]])
		q_end=Quaternion(matrix=curve_R[my_pwlf.break_points[i+1]])
		for j in range(my_pwlf.break_points[i],my_pwlf.break_points[i+1]):
			q_temp=Quaternion.slerp(q_start, q_end, float(j-my_pwlf.break_points[i])/float(my_pwlf.break_points[i+1]-my_pwlf.break_points[i])) 
			curve_R_pred.append(q_temp.rotation_matrix)
			print(curve_direction_ref[i],curve_R_pred[-1])
			curve_final_projection[i]=LinePlaneCollision(planeNormal=curve_direction_ref[i], planePoint=curve[i], rayDirection=curve_R_pred[-1][:,-1], rayPoint=curve_cartesian_pred[i])

	####################################error checking####################################################
	print('calcualting final error')
	max_cartesian_error,max_orientation_error=complete_points_check(curve_final_projection,curve,curve_R_pred,curve_R)
	print('maximum cartesian error: ',max_cartesian_error)
	print('max orientation error: ', max_orientation_error)
	print('average error: ',calc_avg_error(curve_final_projection,curve))
	
if __name__ == "__main__":
	main()