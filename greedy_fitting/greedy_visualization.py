import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import *
import sys, traceback
import numpy as np

from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *

def main():
	dataset='wood/'
	solution_dir='baseline/'
	data_dir="../data/"+dataset+solution_dir
	cmd_dir="../data/"+dataset+solution_dir+'30L/'

	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values

	robot=abb6640(d=50)
	ms=MotionSend()
	breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd(cmd_dir+'command.csv')

	curve_fit, curve_R_fit, curve_js_fit, _=form_traj_from_bp(q_bp,primitives,robot)

	error,angle_error=calc_all_error_w_normal(curve_fit,curve[:,:3],curve_R_fit[:,:,-1],curve[:,3:])

	print(max(error))

	p_bp_np=np.array([item[-1] for item in p_bp])
	plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot(curve[:,0],curve[:,1],curve[:,2],label='Spatial Curve')
	ax.plot(curve_fit[:,0],curve_fit[:,1],curve_fit[:,2],label='Equally Spaced MoveL')
	ax.scatter(p_bp_np[:,0],p_bp_np[:,1],p_bp_np[:,2],label='breakpoints')
	plt.legend()
	

	plt.show()


	# ###plane projection visualization
	# curve_mean = curve[:,:3].mean(axis=0)
	# curve_centered = curve[:,:3] - curve_mean
	# U,s,V = np.linalg.svd(curve_centered)
	# # Normal vector of fitting plane is given by 3rd column in V
	# # Note linalg.svd returns V^T, so we need to select 3rd row from V^T
	# normal = V[2,:]

	# curve_2d_vis = rodrigues_rot(curve_centered+curve_mean, normal, [0,0,1])[:,:2]
	# curve_fit_2d_vis = rodrigues_rot(curve_fit, normal, [0,0,1])[:,:2]
	# plt.plot(curve_2d_vis[:,0],curve_2d_vis[:,1])
	# plt.plot(curve_fit_2d_vis[:,0],curve_fit_2d_vis[:,1])
	# plt.scatter(curve_fit_2d_vis[breakpoints.astype(int),0],curve_fit_2d_vis[breakpoints.astype(int),1])
	# plt.legend(['original curve','curve fit','breakpoints'])
	

	# plt.show()


if __name__ == "__main__":
	main()