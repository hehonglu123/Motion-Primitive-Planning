import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('../toolbox')
from robots_def import *
from utils import *


def main():
	robot=abb6640(d=50)
	data_dir='from_NX/'

	
	col_names=['X', 'Y', 'Z','normal_x','normal_y','normal_z'] 
	data = read_csv(data_dir+'Curve_in_base_frame.csv', names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_normal_x=data['normal_x'].tolist()
	curve_normal_y=data['normal_y'].tolist()
	curve_normal_z=data['normal_z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_normal=np.vstack((curve_normal_x, curve_normal_y, curve_normal_z)).T

	# col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	# data = read_csv(data_dir+'Curve_js.csv', names=col_names)
	# # data = read_csv('../greedy_fitting/curve_fit_js.csv',names=col_names)
	# curve_q1=data['q1'].tolist()
	# curve_q2=data['q2'].tolist()
	# curve_q3=data['q3'].tolist()
	# curve_q4=data['q4'].tolist()
	# curve_q5=data['q5'].tolist()
	# curve_q6=data['q6'].tolist()
	# curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T
	# curve=[]
	# curve_normal=[]
	# for i in range(len(curve_js)):
	# 	pose_temp=robot.fwd(curve_js[i])
	# 	curve.append(pose_temp.p)
	# 	curve_normal.append(pose_temp.R[:,-1])
	# curve=np.array(curve)
	# curve_normal=np.array(curve_normal)

	visualize_curve_w_normal(curve,curve_normal,stepsize=1000,equal_axis=True)

	print(np.amax(curve[:,0])-np.amin(curve[:,0]),np.amax(curve[:,1])-np.amin(curve[:,1]),np.amax(curve[:,2])-np.amin(curve[:,2]))

if __name__ == "__main__":
	main()