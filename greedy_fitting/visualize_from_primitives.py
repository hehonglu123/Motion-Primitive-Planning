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
	fitting_output="../data/"+dataset+'baseline/100L/'
	data = read_csv(fitting_output+'command.csv')
	breakpoints=np.array(data['breakpoints'].tolist())
	primitives=data['primitives'].tolist()
	points=data['points'].tolist()
	
	breakpoints[1:]=breakpoints[1:]-1
	curve_fit_js=read_csv(fitting_output+'curve_fit_js.csv',header=None).values

	q_bp=[]
	for i in range(len(primitives)):
		if primitives[i]=='movej_fit' or primitives[i]=='movel_fit':
			q_bp.append([curve_fit_js[breakpoints[i]]])
		else:
			q_bp.append([curve_fit_js[int((breakpoints[i]+breakpoints[i-1])/2)],curve_fit_js[breakpoints[i]]])

	curve, curve_R, curve_js, breakpoints_new=form_traj_from_bp(curve_fit_js[breakpoints],primitives,robot)

	curve_js_blended,curve_blended,curve_R_blended=blend_js_from_primitive(curve, curve_js, breakpoints_new, primitives,robot,zone=10)

	visualize_curve(curve_blended)
if __name__ == '__main__':
	main()
