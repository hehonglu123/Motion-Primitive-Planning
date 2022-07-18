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

	###initialize
	lam_data=np.arange(len(curve))
	my_pwlf=MDFit(lam_data,curve)
	breakpoints=[0,len(lam_data)-1]
	max_error=999

	while max_error>1:

		if (max_error!=999):

			breakpoints.append(next_breakpoint)
			breakpoints.sort()
		my_pwlf.fit_with_breaks(breakpoints)

		###check error
		fit=my_pwlf.predict()
		max_error=my_pwlf.calc_max_error1()

		###find next breakpoint
		temp_fit=[]
		for i in range(len(breakpoints)-1):
			if i!=len(breakpoints)-2:
				temp_lam=np.arange(breakpoints[i],breakpoints[i+1])
			else:
				temp_lam=np.arange(breakpoints[i],breakpoints[i+1]+1)
			interp=interp1d(np.array([breakpoints[i],breakpoints[i+1]]),np.array([curve[breakpoints[i]],curve[breakpoints[i+1]]]),axis=0)
			temp_fit.append(interp(temp_lam))

		temp_fit=np.concatenate( temp_fit, axis=0 ).reshape(len(curve),len(curve[0]))
		errors=np.linalg.norm(temp_fit-curve,axis=1)
		next_breakpoint=np.argsort(errors)[-1]
		print(len(breakpoints))

	print(max_error)

	
if __name__ == "__main__":
	main()