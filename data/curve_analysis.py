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
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("from_Jon/qbestcurve_new.csv", names=col_names)
	# data = read_csv("from_ge/Curve_js2.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

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
			# theta.append(np.dot(curve[i]-curve[i-1],-curve_R[i-1][:,0]))

	lam=calc_lam_cs(curve)
	print(lam)
	print(theta)

	plt.figure()
	plt.title('theta vs lambda')
	plt.xlabel('lambda (mm)')
	plt.ylabel('theta (rad)')
	plt.plot(lam[:-1],theta)
	plt.show()

if __name__ == "__main__":
	main()