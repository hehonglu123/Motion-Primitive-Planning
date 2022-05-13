import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *


sys.path.append('../toolbox')
from robots_def import *
from lambda_calc import *
from utils import *

def main():

	###read interpolated curves in joint space
	curve_js = read_csv("wood/Curve_js.csv",header=None).values


	robot=abb6640(d=50)
	curve=[]
	curve_R=[]
	theta=[]
	for i in range(len(curve_js)):
		pose=robot.fwd(curve_js[i])
		curve.append(pose.p)
		curve_R.append(pose.R)
		if i>0:
			theta.append(get_angle(curve[i]-curve[i-1],-curve_R[i-1][:,0]))

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
	plt.show()

if __name__ == "__main__":
	main()