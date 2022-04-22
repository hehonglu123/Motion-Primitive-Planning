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

	###read interpolated curves in joint space
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("from_ge/Curve_js2.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T


	abb6640_obj=abb6640(R_tool=np.eye(3),p_tool=np.zeros(3),d=0)

	curve_R=[]

	curve=[]
	for i in range(len(curve_js)):
		pose=abb6640_obj.fwd(curve_js[i])
		curve.append(pose.p)
		curve_R.append(pose.R)

	curve=np.array(curve)
	curve_R=np.array(curve_R)


	###output to csv
	df=DataFrame({'x':curve[:,0],'y':curve[:,1],'z':curve[:,2],'R11':curve_R[:,0,0],'R12':curve_R[:,0,1],'R13':curve_R[:,0,2],'R21':curve_R[:,1,0],'R22':curve_R[:,1,1],'R23':curve_R[:,1,2],'R31':curve_R[:,2,0],'R32':curve_R[:,2,1],'R33':curve_R[:,2,2]})
	df.to_csv('from_ge/Curve_fwd_eef.csv',header=True,index=False)


if __name__ == "__main__":
	main()