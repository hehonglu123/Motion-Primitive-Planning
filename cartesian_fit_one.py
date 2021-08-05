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
	data = read_csv("data/Curve_dense.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T

	###x ref, yz out
	my_pwlf=MDFit(np.arange(len(curve)),curve)

	###slope calc breakpoints
	my_pwlf.break_slope_simplified(-1)
	print(len(my_pwlf.break_points))
	my_pwlf.fit_with_breaks(my_pwlf.x_data[my_pwlf.break_points])

	###predict for the determined points
	xHat = np.arange(len(curve))
	pred = my_pwlf.predict_arb(xHat)

	print('maximum error: ',np.max(np.linalg.norm(pred-curve,axis=1)))
	print('average error: ',np.average(np.linalg.norm(pred-curve,axis=1)))
	
if __name__ == "__main__":
	main()