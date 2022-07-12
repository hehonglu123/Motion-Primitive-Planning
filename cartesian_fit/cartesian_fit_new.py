import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import *
from pwlfmd import *
import sys
import numpy as np
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
	max_seg=50

	while max_error>1:

		if (max_error!=999):

			breakpoints.append(next_breakpoint)
			breakpoints.sort()
		my_pwlf.fit_with_breaks(breakpoints)

		###check error
		fit=my_pwlf.predict()
		errors=np.linalg.norm(fit-curve,axis=1)
		max_error=np.max(errors)
		idx_sort=np.argsort(errors)
		for i in range(len(idx_sort)-1,0,-1):
			if idx_sort[i] not in breakpoints and np.min(np.abs(breakpoints-idx_sort[i]))>len(curve)/max_seg:
				next_breakpoint=idx_sort[i]
				break
		if i==0:
			break
		print(len(breakpoints))
	print(max_error)

	
if __name__ == "__main__":
	main()