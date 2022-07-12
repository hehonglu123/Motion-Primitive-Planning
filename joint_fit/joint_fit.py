import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import *
from pwlfmd import *
import numpy as np
import sys
sys.path.append('toolbox')
from error_check import *
from robot_def import *

def main():
	###read actual curve
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("train_data/Curve_dense.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T

	###read interpolated curves in joint space
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("train_data/Curve_js.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	###x ref, yz out
	my_pwlf=MDFit(np.arange(len(curve_js)),curve_js)

	###slope calc breakpoints
	# break_points=my_pwlf.lam_data[my_pwlf.break_slope()]
	# my_pwlf.fit_with_breaks(break_points)

	###fit by error thresholding
	my_pwlf.fit_under_error_simplified(max_error=0.02,starting_threshold=0.005)

	###predict at every train_data index
	xHat = np.arange(len(curve_js))
	curve_js_pred = my_pwlf.predict_arb(xHat)

	curve_cartesian_pred=[]
	curve_R_pred=[]
	for q in curve_js_pred:
		fwdkin_result=fwd(q)
		curve_cartesian_pred.append(fwdkin_result.p)
		curve_R_pred.append(fwdkin_result.R)
	curve_R=[]
	for q in curve_js:
		curve_R.append(fwd(q).R)
	###units
	curve_cartesian_pred=np.array(curve_cartesian_pred)
	###convert to reference frame
	R=np.array([[0,0,1.],
			[1.,0,0],
			[0,1.,0]])
	T=np.array([[2700.],[-800.],[500.]])
	H=np.vstack((np.hstack((R.T,-np.dot(R.T,T))),np.array([0,0,0,1])))
	for i in range(len(curve_cartesian_pred)):
		curve_cartesian_pred[i]=np.dot(H,np.hstack((curve_cartesian_pred[i],[1])).T)[:-1]

	print('calcualting final error')
	max_cartesian_error,max_orientation_error=complete_points_check(curve_cartesian_pred,curve,curve_R_pred,curve_R)
	print('maximum cartesian error: ',max_cartesian_error)
	print('max orientation error: ', max_orientation_error)
	print('average error: ',calc_avg_error(curve_cartesian_pred,curve))

if __name__ == "__main__":
	main()