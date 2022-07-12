import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import *
from pwlfmd import *
import sys
import numpy as np
from scipy.interpolate import interp1d
sys.path.append('toolbox')
from error_check import *


def main():
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("train_data/from_interp/Curve_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T

	###read interpolated curves in joint space
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("train_data/from_interp/Curve_js.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	###initialize
	lam_data=np.arange(len(curve_js))
	my_pwlf=MDFit(lam_data,curve_js)
	breakpoints=[0,len(lam_data)-1]
	max_error=999

	while max_error>1:

		if (max_error!=999):

			breakpoints.append(next_breakpoint)
			breakpoints.sort()
		my_pwlf.fit_with_breaks(breakpoints)

		###check error
		fit=my_pwlf.predict()
		fit_in_cartesian=[]
		for i in range(len(curve_js)):
			fit_in_cartesian.append(fwd(fit[i]).p)
		max_error=np.max(np.linalg.norm(curve-np.array(fit_in_cartesian),axis=1))
		print(max_error)
		###find next breakpoint
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
		print(len(breakpoints))

	print(max_error)

	
if __name__ == "__main__":
	main()