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
	data = read_csv("train_data/Curve_dense.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T

	###x ref, yz out
	my_pwlf=MDFit(np.arange(len(curve)),curve)

	###arbitrary breakpoints
	# break_points=[np.min(curve[:,0]),500,1000,np.max(curve[:,0])]

	###slope calc breakpoints
	# break_points=my_pwlf.lam_data[my_pwlf.break_slope()]
	# my_pwlf.fit_with_breaks(break_points)

	###fit by error thresholding
	my_pwlf.fit_under_error_simplified(1)

	###predict for the determined points
	xHat = np.linspace(0,len(curve), num=1000)
	pred = my_pwlf.predict_arb(xHat)

	# curve_fit=pred

	# # print('maximum error: ',calc_max_error(curve_fit,curve))
	# # print('average error: ',calc_avg_error(curve_fit,curve))

	# ###plot results
	# fig = plt.figure()
	# ax = plt.axes(projection='3d')
	# ax.plot3D(pred[:,0], pred[:,1], pred[:,2], 'gray')
	# ax.scatter3D(curve_x, curve_y, curve_z, c=curve_z, cmap='Accent');

	# plt.show()

	print('maximum error: ',calc_max_error(pred,curve))
	print('average error: ',calc_avg_error(pred,curve))
	
if __name__ == "__main__":
	main()