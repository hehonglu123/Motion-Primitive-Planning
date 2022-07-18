import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from pandas import *
from fitting_toolbox import *
import sys
sys.path.append('../toolbox')
from toolbox_circular_fit import *
from robots_def import *
from general_robotics_toolbox import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *


def main():
	robot=abb6640(d=50)

	dataset='wood/'
	# fitting_output="../train_data/"+dataset+'baseline/100L/'
	fitting_output= 'greedy_output/'
	data = read_csv(fitting_output+'command.csv')
	breakpoints=np.array(data['breakpoints'].tolist())
	primitives=data['primitives'].tolist()
	
	breakpoints[1:]=breakpoints[1:]-1
	curve_fit_js=read_csv(fitting_output+'curve_fit_js.csv',header=None).values

	q_bp=[]
	for i in range(len(primitives)):
		if primitives[i]=='movej_fit' or primitives[i]=='movel_fit':
			q_bp.append([curve_fit_js[breakpoints[i]]])
		else:
			q_bp.append([curve_fit_js[int((breakpoints[i]+breakpoints[i-1])/2)],curve_fit_js[breakpoints[i]]])


	curve, curve_R, curve_js, breakpoints_new=form_traj_from_bp(q_bp,primitives,robot)

	curve_js_blended,curve_blended,curve_R_blended=blend_js_from_primitive(curve, curve_js, breakpoints_new, primitives,robot,zone=10)

	print(primitives)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'gray', label='interploation')
	ax.plot3D(curve_blended[:,0], curve_blended[:,1],curve_blended[:,2], 'green', label='blended trajectroy')
	ax.scatter(curve[breakpoints_new,0], curve[breakpoints_new,1],curve[breakpoints_new,2], 'gray',label='breakpoints')
	ax.set_xlabel('x (mm)')
	ax.set_ylabel('y (mm)')
	ax.set_zlabel('z (mm)')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
