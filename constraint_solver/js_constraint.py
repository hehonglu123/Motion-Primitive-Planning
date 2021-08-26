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
	data = read_csv("../data/original/Curve_js.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	joint_vel_limit=np.radians([110,90,90,150,120,235])
	joint_acc_limit=10*np.ones(6)

	dlam_max=[]
	ddlam_max=[]
	dlam_act=[0]
	qd_prev=np.zeros(6)
	resolution=100

	for lam in range(len(curve_js)-1):
		dq=np.abs(curve_js[lam+1]-curve_js[lam])


		t=np.max(dq/joint_vel_limit)
		qd_max=dq/t
		q_prime=dq/1				
		dlam_max.append(qd_max[0]/q_prime[0])
		###acc constraint
		
		t2qd_max=np.max(np.abs(qd_max-qd_prev)/joint_acc_limit)
		qdd_max=qd_max/t2qd_max

		qd_prev=qd_max

		ddlam_max.append(qdd_max[0]/q_prime[0])

		for r in range(resolution-1):
			if dlam_act[-1]!=dlam_max[-1]:
				dlam_act.append(dlam_act[-1]+np.clip(dlam_max[-1]-dlam_act[-1],-np.abs(ddlam_max[-1]/(resolution-1)),np.abs(ddlam_max[-1]/(resolution-1))))
			else:
				dlam_act.append(dlam_max[-1])

	dlam_act.pop(0)
	lam=np.arange(0,len(curve_js)-1,1./(resolution-1))
	plt.plot(lam,np.repeat(dlam_max,(resolution-1)),label="lambda_dot_max")
	plt.plot(lam,dlam_act,label="lambda_dot_act")
	plt.legend()
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.title("max lambda_dot vs lambda (path index)")
	plt.savefig("velocity-constraint_js.png")
	plt.show()


if __name__ == "__main__":
	main()