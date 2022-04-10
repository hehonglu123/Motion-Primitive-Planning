from pandas import read_csv, DataFrame
import sys, copy
sys.path.append('../circular_Fit')
from toolbox_circular_fit import *
from abb_motion_program_exec_client import *
from robots_def import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import UnivariateSpline
from lambda_calc import *
from blending import *

def main():

	data_dir='../simulation/robotstudio_sim/scripts/fitting_output_new/threshold0.5/'

	data = read_csv(data_dir+'command.csv')
	breakpoints=np.array(data['breakpoints'].tolist())#[:3]
	primitives=data['primitives'].tolist()[1:]#[:2]

	col_names=['J1', 'J2','J3', 'J4', 'J5', 'J6'] 
	data=read_csv(data_dir+'curve_fit_js.csv',names=col_names)
	q1=data['J1'].tolist()
	q2=data['J2'].tolist()
	q3=data['J3'].tolist()
	q4=data['J4'].tolist()
	q5=data['J5'].tolist()
	q6=data['J6'].tolist()
	curve_js=np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)

	data = read_csv(data_dir+'curve_fit.csv')
	curve_x=data['x'].tolist()
	curve_y=data['y'].tolist()
	curve_z=data['z'].tolist()
	curve_fit=np.vstack((curve_x, curve_y, curve_z)).T

	robot=abb6640(d=50)



	act_breakpoints=breakpoints
	act_breakpoints[1:]=act_breakpoints[1:]-1

	lam=calc_lam_cs(curve_fit)[:act_breakpoints[-1]]

	lam_blended,q_blended=blend_cs(curve_js[act_breakpoints],curve_fit,breakpoints,lam,primitives,robot)

	curve_blended=[]
	for i in range(len(q_blended)):
		curve_blended.append(robot.fwd(q_blended[i]).p)
	curve_blended=np.array(curve_blended)


	###plot results
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot3D(curve_fit[:act_breakpoints[-1],0], curve_fit[:act_breakpoints[-1],1], curve_fit[:act_breakpoints[-1],2], c='gray')
	ax.plot3D(curve_blended[:,0], curve_blended[:,1], curve_blended[:,2], c='red')
	ax.scatter(curve_fit[breakpoints,0], curve_fit[breakpoints,1], curve_fit[breakpoints,2])

	plt.show()

if __name__ == '__main__':
	main()
