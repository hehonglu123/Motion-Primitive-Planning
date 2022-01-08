import numpy as np
from pandas import *
import sys, traceback, time
from general_robotics_toolbox import *
import matplotlib.pyplot as plt


sys.path.append('../')
from constraint_solver import *

def main():
	joint_vel_limit=np.radians([110,90,90,150,120,235])
	base2_R=np.array([[-1,0,0],[0,-1,0],[0,0,1]])
	base2_p=np.array([6000,0,0])
	###read actual curve
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("trajectory/arm1.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js1=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("trajectory/arm2.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js2=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T


	p_prev=np.zeros(3)
	###find path length
	lam=[0]
	for i in range(len(curve_js1)-1):
		p_new=fwd(curve_js1[i+1]).p-fwd(curve_js2[i+1],base2_R,base2_p).p
		lam.append(lam[-1]+np.linalg.norm(p_new-p_prev))
		p_prev=p_new
	###normalize lam, 
	lam=np.array(lam)/lam[-1]


	dlam_out=calc_lamdot(np.hstack((curve_js1,curve_js2)),lam[:len(curve_js2)],np.tile(joint_vel_limit,2),1)


	plt.plot(lam[:-1],dlam_out,label="lambda_dot_max")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.title("max lambda_dot vs lambda (path index)")
	plt.show()

if __name__ == "__main__":
	main()