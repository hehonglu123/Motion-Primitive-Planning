import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *

from robots_def import *
from lambda_calc import *
from utils import *

def main():
	data_dir='wood/'

	###read interpolated curves in joint space
	curve_js1 = read_csv(data_dir+'dual_arm/diffevo2/arm1.csv',header=None).values
	curve_js2 = read_csv(data_dir+'dual_arm/diffevo2/arm2.csv',header=None).values


	robot1=abb6640(d=50)

	with open(data_dir+'dual_arm/tcp.yaml') as file:
	    H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
	robot2=abb1200(R_tool=H_tcp[:3,:3],p_tool=H_tcp[:-1,-1])

	robot=robot2
	curve_js=curve_js2

	curve=[]
	curve_R=[]
	theta=[]
	for i in range(len(curve_js)):
		pose=robot.fwd(curve_js[i])
		curve.append(pose.p)
		curve_R.append(pose.R)
		if i>0:
			theta.append(get_angle(curve[i]-curve[i-1],-curve_R[i-1][:,0]))

	J_det=find_j_det(robot,curve_js)
	J_cond=find_condition_num(robot,curve_js)
	J_sing_min=find_j_min(robot,curve_js)

	lam=calc_lam_cs(curve)


	plt.figure()
	plt.title('theta vs lambda')
	plt.xlabel('lambda (mm)')
	plt.ylabel('theta (rad)')
	plt.plot(lam[1:],theta)

	plt.figure()
	plt.plot(lam,J_sing_min)
	plt.title('Min Jacobian Singular')

	plt.figure()
	plt.plot(lam,J_det)
	plt.title('Jacobian Determinant')

	plt.figure()
	plt.plot(lam,J_cond)
	plt.title('Jacobian Condition Number')
	plt.show()

if __name__ == "__main__":
	main()