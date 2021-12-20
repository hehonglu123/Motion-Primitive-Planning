import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt

sys.path.append('../toolbox')
from robot_def import *
from lambda_calc import *

def main():
	###read actual curve
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("curve_poses/curve_pose6/Curve_backproj_js1.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	joint_vel_limit=np.radians([110,90,90,150,120,235])


	dlam_max=[]

	###find path length
	lam=[0]
	for i in range(len(curve_js)-1):
		ps=fwd(curve_js[i]).p
		pe=fwd(curve_js[i+1]).p
		lam.append(lam[-1]+np.linalg.norm(pe-ps))

	###normalize lam
	lam=np.array(lam)/lam[-1]

	step=1000

	dlam_max=calc_lamdot(curve_js,lam,joint_vel_limit,step=1000)

	plt.plot(lam[:-step:step],dlam_max,label="lambda_dot_max")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.title("max lambda_dot vs lambda (path index)")
	plt.savefig("velocity-constraint_js.png")
	plt.show()


if __name__ == "__main__":
	main()