import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from pandas import *
import sys,copy
sys.path.append('../circular_fit')
from toolbox_circular_fit import *
sys.path.append('../toolbox')
from robot_def import *

def movel_fit(curve,p=[]):
	###no constraint
	if len(p)==0:
		A=np.vstack((np.ones(len(curve)),np.arange(0,len(curve)))).T
		b=curve
		res=np.linalg.lstsq(A,b,rcond=None)[0]
		start_point=res[0]
		slope=res[1].reshape(1,-1)
	###with constraint point
	else:
		A=np.arange(0,len(curve)).reshape(-1,1)
		b=curve-curve[0]
		res=np.linalg.lstsq(A,b,rcond=None)[0]
		slope=res.reshape(1,-1)
		start_point=p

	curve_fit=np.dot(np.arange(0,len(curve)).reshape(-1,1),slope)+start_point

	max_error=np.max(np.linalg.norm(curve-curve_fit,axis=1))

	return curve_fit,max_error



def movec_fit(curve,p=[]):

	curve_fit,curve_fit_circle=circle_fit(curve,p)
	max_error=np.max(np.linalg.norm(curve-curve_fit,axis=1))

	return curve_fit,max_error


def fit_w_breakpoints(curve,primitives_choices,breakpoints):
	curve_fit=[]
	max_error_all=0
	for i in range(len(primitives_choices)):
		if i==0:
			fit_output,max_error=primitives_choices[i](curve[breakpoints[i]:breakpoints[i+1]])
		else:
			fit_output,max_error=primitives_choices[i](curve[breakpoints[i]:breakpoints[i+1]],curve_fit[-1])
		curve_fit.extend(fit_output)
		if max_error>max_error_all:
			max_error_all=copy.deepcopy(max_error)

	curve_fit=np.array(curve_fit)
	return curve_fit, max_error_all

def main():
	###read in points
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../data/from_cad/Curve_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T


	curve_fit,max_error_all=fit_w_breakpoints(curve,[movec_fit,movel_fit,movec_fit],[0,int(len(curve)/3),int(2*len(curve)/3),len(curve)])

	print(max_error_all)

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot3D(curve[:,0], curve[:,1], curve[:,2], 'gray')
	ax.plot3D(curve_fit[:,0], curve_fit[:,1], curve_fit[:,2], 'red')

	plt.show()
if __name__ == "__main__":
	main()
