import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt

sys.path.append('../toolbox')
from robot_def import *


def main():
	###read actual curve
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("../data/execution/Curve_moveJ.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	joint_vel_limit=np.radians([110,90,90,150,120,235])
	joint_acc_limit=10*np.ones(6)

	breakpoints=[0, 422, 1180, 1927, 2650, 3349, 4022, 4671, 5308, 5961, 6696, 7575, 9105, 10703, 11966, 13170, 14363, 15571, 16808, 18087, 19412, 20780, 22192, 23551, 24883, 26244, 27647, 29106, 30638, 32249, 33898, 34946]


	dlam_max=[]
	ddlam_max=[]
	dlam_act=[0]
	qd_prev=np.zeros(6)

	###find path length
	lam=[0]
	for i in range(len(curve_js)-1):
		ps=fwd(curve_js[i]).p
		pe=fwd(curve_js[i+1]).p
		lam.append(lam[-1]+np.linalg.norm(pe-ps))

	###normalize lam
	lam=np.array(lam)/lam[-1]


	for i in range(len(curve_js)-1):
		dq=np.abs(curve_js[i+1]-curve_js[i])
		q_prime=dq/(lam[i+1]-lam[i])
		t=np.max(dq/joint_vel_limit)
		qd_max=dq/t 		###approximated max qdot
		dlam_max.append(qd_max[0]/q_prime[0])


	dlam_act.pop(0)
	plt.plot(lam[1:],dlam_max,label="lambda_dot_max")
	# plt.plot(lam,dlam_act,label="lambda_dot_act")
	plt.legend()
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.title("max lambda_dot vs lambda (path index)")
	plt.savefig("velocity-constraint_js.png")
	plt.show()


if __name__ == "__main__":
	main()