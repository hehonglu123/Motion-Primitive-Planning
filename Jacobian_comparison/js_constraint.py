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
	data = read_csv("curve_poses/curve_pose0/Curve_backproj_js0.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	joint_vel_limit=np.radians([110,90,90,150,120,235])


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


	for i in range(0,100):#len(curve_js)-1,1):
		dq=np.abs(curve_js[i+1]-curve_js[i])
		dqdlam=dq/(lam[i+1]-lam[i])
		t=np.max(dq/joint_vel_limit)

		qdot_max=dq/t 		###approximated max qdot
		dlam_max.append(qdot_max[0]/dqdlam[0])

		print(t,np.linalg.norm(dq))



	dlam_act.pop(0)
	# plt.plot(lam[1:-1:10],dlam_max,label="lambda_dot_max")
	# plt.xlabel("lambda")
	# plt.ylabel("lambda_dot")
	# plt.title("max lambda_dot vs lambda (path index)")
	# plt.savefig("velocity-constraint_js.png")
	# plt.show()


if __name__ == "__main__":
	main()