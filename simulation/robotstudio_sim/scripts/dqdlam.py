import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import *
import sys, traceback
import numpy as np
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *

def main():
	data_dir="fitting_output/slope_blend/"
	col_names=['J1', 'J2','J3', 'J4', 'J5', 'J6'] 
	data = read_csv(data_dir+"curve_fit_js.csv",names=col_names)

	q1=data['J1'].tolist()[1:-1]
	q2=data['J2'].tolist()[1:-1]
	q3=data['J3'].tolist()[1:-1]
	q4=data['J4'].tolist()[1:-1]
	q5=data['J5'].tolist()[1:-1]
	q6=data['J6'].tolist()[1:-1]
	q_all=np.vstack((q1,q2,q3,q4,q5,q6)).T

	robot=abb6640()
	curve=[]
	lam=[0]
	dq_dlam=[]
	d2q_dlam2=[]
	for i in range(len(q_all)):
		q=np.radians(q_all[i])
		curve.append(robot.fwd(q).p)
		if i>0:
			lam.append(lam[-1]+np.linalg.norm(curve[i]-curve[i-1]))
			dq_dlam.append((q_all[i]-q_all[i-1])/(lam[-1]-lam[-2]))
			if i>1:
				d2q_dlam2.append((dq_dlam[-1]-dq_dlam[-2])/(lam[-1]-lam[-2]))
		
	dq_dlam=np.array(dq_dlam)
	d2q_dlam2=np.array(d2q_dlam2)
	df=DataFrame({'j1':dq_dlam[:,0],'j2':dq_dlam[:,1],'j3':dq_dlam[:,2],'j4':dq_dlam[:,3],'j5':dq_dlam[:,4],'j6':dq_dlam[:,5]})
	df.to_csv(data_dir+'dqdlam.csv',header=True,index=False)
	df=DataFrame({'j1':d2q_dlam2[:,0],'j2':d2q_dlam2[:,1],'j3':d2q_dlam2[:,2],'j4':d2q_dlam2[:,3],'j5':d2q_dlam2[:,4],'j6':d2q_dlam2[:,5]})
	df.to_csv(data_dir+'d2qdlam2.csv',header=True,index=False)

if __name__ == "__main__":
	main()