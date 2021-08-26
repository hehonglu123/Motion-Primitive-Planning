import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt

sys.path.append('../toolbox')
from robot_def import *

def find_dlam_max(qd_prev,qs,qe):
	global joint_vel_limit, resolution
	ps=fwd(qs)
	pe=fwd(qe)

	R_diff=np.dot(ps.R.T,pe.R)
	k,theta=R2rot(R_diff)
	k=np.array(k)
	moving_direction=np.hstack((theta*k,pe.p-ps.p))

	dlam_max=[]
	ddlam_max=[]
	dlam_act=[0]
	resolution=100

	for r in range(resolution-1):
		###interpolation
		p_temp=((resolution-r)*ps.p+r*pe.p)/resolution
		theta_temp=r*theta/resolution
		R_temp=np.dot(ps.R,q2R(rot2q(k,theta_temp)))

		###solve best inv config
		q_all=np.array(inv(p_temp,R_temp))

		###choose inv_kin closest to previous joints
		try:
			temp_q=q_all-qs
			order=np.argsort(np.linalg.norm(temp_q,axis=1))
			q_interp=q_all[order[0]]

		except:
			traceback.print_exc()
			pass

		J=jacobian(q_interp)
		dlam_max.append(np.min(joint_vel_limit/np.abs(np.dot(np.linalg.inv(J),moving_direction))))

	return dlam_max

def main():
	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = read_csv("../data/original/Curve_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z))

	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("../data/original/Curve_js.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	global joint_vel_limit, resolution
	joint_vel_limit=np.radians([110,90,90,150,120,235])
	dlam_max=[]

	resolution=100
	for lam in range(len(curve_js)-1):
		dlam_max+=find_dlam_max(curve_js[lam],curve_js[lam+1])


	lam=np.arange(0,len(curve_js)-1,1./(resolution-1))
	plt.plot(lam,dlam_max)
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.title("lambda_dot vs lambda (path index)")
	plt.savefig("velocity-constraint_cs.png")
	plt.show()



if __name__ == "__main__":
	main()