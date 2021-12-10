import numpy as np
from pandas import *
import sys, traceback, time
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import differential_evolution, shgo, NonlinearConstraint
from qpsolvers import solve_qp

sys.path.append('../toolbox')
from robot_def import *
from lambda_calc import *

def normalize_dq(q):
	q=q/(np.linalg.norm(q)) 
	return q   

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

def opt_fun(theta0,lam,joint_vel_limit,curve,curve_normal):

	R_temp=direction2R(curve_normal[0],-curve[1]+curve[0])
	R=np.dot(R_temp,Rz(theta0[0]))
	try:
		q_init=inv(curve[0],R)[0]
		q_out=stepwise_optimize(q_init,curve,curve_normal)
	except:
		# traceback.print_exc()
		return 999

	
	dlam=calc_lamdot(q_out,lam,joint_vel_limit,1)
	print(min(dlam))
	return -min(dlam)

def stepwise_optimize(q_init,curve,curve_normal):
	q_all=[q_init]
	q_out=[q_init]
	for i in range(len(curve)):
		try:
			error_fb=999
			if i==0:
				continue

			while error_fb>0.1:

				pose_now=fwd(q_all[-1])
				error_fb=np.linalg.norm(pose_now.p-curve[i])+np.linalg.norm(pose_now.R[:,-1]-curve_normal[i])
				
				w=0.2
				Kq=.01*np.eye(6)    #small value to make sure positive definite
				KR=np.eye(3)        #gains for position and orientation error

				J=jacobian(q_all[-1])        #calculate current Jacobian
				Jp=J[3:,:]
				JR=J[:3,:] 
				H=np.dot(np.transpose(Jp),Jp)+Kq+w*np.dot(np.transpose(JR),JR)
				H=(H+np.transpose(H))/2

				vd=curve[i]-pose_now.p
				k=np.cross(pose_now.R[:,-1],curve_normal[i])

				k=k/np.linalg.norm(k)
				theta=-np.arctan2(np.linalg.norm(np.cross(pose_now.R[:,-1],curve_normal[i])), np.dot(pose_now.R[:,-1],curve_normal[i]))

				k=np.array(k)
				s=np.sin(theta/2)*k         #eR2
				wd=-np.dot(KR,s)
				f=-np.dot(np.transpose(Jp),vd)-w*np.dot(np.transpose(JR),wd)
				qdot=solve_qp(H,f)

				q_all.append(q_all[-1]+qdot)
				# print(q_all[-1])
		except:
			q_out.append(q_all[-1])
			traceback.print_exc()
			raise AssertionError
			break

		q_out.append(q_all[-1])

	q_out=np.array(q_out)
	return q_out

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
	bnds=tuple(zip(lowerer_limit,upper_limit))


	res = differential_evolution(opt_fun, bnds, args=(lam,joint_vel_limit,curve_cs_p,curve_cs_R[:,:,-1],),workers=-1,
									x0 = [0],
									strategy='best1bin', maxiter=2000,
									popsize=15, tol=1e-10,
									mutation=(0.5, 1), recombination=0.7,
									seed=None, callback=None, disp=False,
									polish=True, init='latinhypercube',
									atol=0.)
	


	print(res)

	R_temp=direction2R(curve_cs_R[0,:,-1],-curve_cs_p[1]+curve_cs_p[0])
	R=np.dot(R_temp,Rz(res.x[0]))
	q_init=inv(curve_cs_p[0],R)[0]
	q_out=stepwise_optimize(q_init,curve_cs_p,curve_cs_R[:,:,-1])

	dlam_out=calc_lamdot(q_out,lam[:len(q_out)],joint_vel_limit,1)


	plt.plot(lam[:len(q_out)-1],dlam_out,label="lambda_dot_max")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.title("max lambda_dot vs lambda (path index)")
	plt.savefig("velocity-constraint_js.png")
	plt.show()
	

if __name__ == "__main__":
	main()