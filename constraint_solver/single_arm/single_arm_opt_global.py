import numpy as np
from pandas import *
import sys, traceback, time
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import differential_evolution, shgo, NonlinearConstraint

sys.path.append('../toolbox')
from robot_def import *
from lambda_calc import *

def direction2R(v_norm,v_tang):
	v_norm=v_norm/np.linalg.norm(v_norm)
	theta1 = np.arccos(np.dot(np.array([0,0,1]),v_norm))
	###rotation to align z axis with curve normal
	axis_temp=np.cross(np.array([0,0,1]),v_norm)
	R1=rot(axis_temp/np.linalg.norm(axis_temp),theta1)

	###find correct x direction
	v_temp=v_tang-v_norm * np.dot(v_tang, v_norm) / np.linalg.norm(v_norm)

	###get as ngle to rotate
	theta2 = np.arccos(np.dot(R1[:,0],v_temp/np.linalg.norm(v_temp)))


	axis_temp=np.cross(R1[:,0],v_temp)
	axis_temp=axis_temp/np.linalg.norm(axis_temp)

	###rotation about z axis to minimize x direction error
	R2=rot(np.array([0,0,np.sign(np.dot(axis_temp,v_norm))]),theta2)


	return np.dot(R1,R2)

def opt_fun(theta,lam,joint_vel_limit,curve,curve_normal):

	for i in range(len(curve)):
		if i==0:
			R_temp=direction2R(curve_normal[i],-curve[i+1]+curve[i])
			R=np.dot(R_temp,Rz(theta[i]))
			try:
				q_out=[inv(curve[i],R)[0]]
			except:
				traceback.print_exc()
				return 999

		else:
			R_temp=direction2R(curve_normal[i],-curve[i]+curve[i-1])
			R=np.dot(R_temp,Rz(theta[i]))
			try:
				###get closet config to previous one
				q_inv_all=inv(curve[i],R)
				temp_q=q_inv_all-q_out[-1]
				order=np.argsort(np.linalg.norm(temp_q,axis=1))
				q_out.append(q_inv_all[order[0]])
			except:
				# traceback.print_exc()
				return 999


	dlam=calc_lamdot(q_out,lam,joint_vel_limit,1)
	print(min(dlam))
	return -min(dlam)

def calc_out(theta,lam,joint_vel_limit,curve,curve_normal):

	for i in range(len(curve)):
		if i==0:
			R_temp=direction2R(curve_normal[i],-curve[i+1]+curve[i])
			R=np.dot(R_temp,Rz(theta[i]))
			try:
				q_out=[inv(curve[i],R)[0]]
			except:
				traceback.print_exc()
				return 999

		else:
			R_temp=direction2R(curve_normal[i],-curve[i]+curve[i-1])
			R=np.dot(R_temp,Rz(theta[i]))
			try:
				###get closet config to previous one
				q_inv_all=inv(curve[i],R)
				temp_q=q_inv_all-q_out[-1]
				order=np.argsort(np.linalg.norm(temp_q,axis=1))
				q_out.append(q_inv_all[order[0]])
			except:
				# traceback.print_exc()
				return 999


	return calc_lamdot(q_out,lam,joint_vel_limit,1)



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

	###decrease curve density to simplify computation
	step=1000	
	curve_js=curve_js[0:-1:step]

	curve_cs_p=[]
	curve_cs_R=[]
	###find path length
	lam=[0]
	for i in range(len(curve_js)-1):
		pose_s=fwd(curve_js[i])
		pose_e=fwd(curve_js[i+1])
		lam.append(lam[-1]+np.linalg.norm(pose_e.p-pose_s.p))
		curve_cs_p.append(pose_s.p)
		curve_cs_R.append(pose_s.R)

		if i==len(curve_js)-2:
			curve_cs_p.append(pose_e.p)
			curve_cs_R.append(pose_e.R)
	###normalize lam
	lam=np.array(lam)/lam[-1]

	curve_cs_p=np.array(curve_cs_p)
	curve_cs_R=np.array(curve_cs_R)

	###path constraints, position constraint and curve normal constraint
	lowerer_limit=[-np.pi]
	upper_limit=[np.pi]
	bnds=tuple(zip(lowerer_limit,upper_limit))*len(curve_js)

	# res = minimize(opt_fun, np.zeros(len(curve_js)), args=(lam,joint_vel_limit,curve_cs_p,curve_cs_R[:,:,-1],), method='SLSQP',tol=1e-10,bounds=bnds)

	###diff evolution
	res = differential_evolution(opt_fun, bnds, args=(lam,joint_vel_limit,curve_cs_p,curve_cs_R[:,:,-1],),workers=-1,
									x0 = np.zeros(len(curve_js)),
									strategy='best1bin', maxiter=2000,
									popsize=15, tol=1e-2,
									mutation=(0.5, 1), recombination=0.7,
									seed=None, callback=None, disp=False,
									polish=True, init='latinhypercube',
									atol=0.)

	###shgo, doesn't take x0
	# cons = ({'type': 'eq', 'fun': lambda x:  fwd_all(x.reshape((-1,6))).p_all.flatten()-curve_cs_p.flatten()},
	# 		{'type': 'eq', 'fun': lambda x:  fwd_all(x.reshape((-1,6))).R_all[:,:,-1].flatten()-curve_cs_R[:,:,-1].flatten()})
	# res=shgo(opt_fun, bnds, args=(lam,joint_vel_limit,), constraints=None, n=None, iters=1, callback=None, minimizer_kwargs=None, options=None, sampling_method='simplicial')


	###trust-constr not working
	print(res)
	dlam_out=calc_out(res.x,lam,joint_vel_limit,curve_cs_p,curve_cs_R[:,:,-1])


	plt.plot(lam[:-1],dlam_out,label="lambda_dot_max")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.title("max lambda_dot vs lambda (path index)")
	plt.savefig("velocity-constraint_js.png")
	plt.show()
	

if __name__ == "__main__":
	main()