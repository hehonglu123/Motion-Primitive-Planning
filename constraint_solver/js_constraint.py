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
	limiting_joint=[]
	qd_prev=np.zeros(6)

	for i in range(len(breakpoints)-1):

		dq=np.abs(curve_js[breakpoints[i+1]]-curve_js[breakpoints[i]])

		t=np.max(dq/joint_vel_limit)
		limiting_joint.append(np.argmax(dq/joint_vel_limit))
		qd_max=dq/t
		q_prime=dq/((breakpoints[i+1]-breakpoints[i])/len(curve_js))
		dlam_max+=[qd_max[0]/q_prime[0]]*(breakpoints[i+1]-breakpoints[i])
		###acc constraint
		
		t2qd_max=np.max(np.abs(qd_max-qd_prev)/joint_acc_limit)
		qdd_max=qd_max/t2qd_max

		qd_prev=qd_max

		ddlam_max.append(qdd_max[0]/q_prime[0])

		for r in range(breakpoints[i],breakpoints[i+1]):
			if dlam_act[-1]!=dlam_max[-1]:
				dlam_act.append(dlam_act[-1]+np.clip(dlam_max[-1]-dlam_act[-1],-np.abs(ddlam_max[-1]/(breakpoints[i+1]-breakpoints[i])),np.abs(ddlam_max[-1]/(breakpoints[i+1]-breakpoints[i]))))
			else:
				dlam_act.append(dlam_max[-1])

	print(limiting_joint)
	dlam_act.pop(0)
	lam=np.arange(0,len(curve_js)-1)/(len(curve_js))
	plt.plot(lam,dlam_max,label="lambda_dot_max")
	# plt.plot(lam,dlam_act,label="lambda_dot_act")
	plt.legend()
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.title("max lambda_dot vs lambda (path index)")
	plt.savefig("velocity-constraint_js.png")
	plt.show()


if __name__ == "__main__":
	main()