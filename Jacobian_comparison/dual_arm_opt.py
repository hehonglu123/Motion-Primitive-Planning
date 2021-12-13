import numpy as np
from pandas import *
import sys, traceback, time
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sys.path.append('../toolbox')
from robot_def import *
from lambda_calc import *

def stepwise_optimize(q_init1,q_init2,curve,curve_normal):
	q_all1=[q_init1]
	q_out1=[q_init1]
	q_all2=[q_init2]
	q_out2=[q_init2]
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
			break

		q_out.append(q_all[-1])

	q_out=np.array(q_out)
	return q_out


def main():
	global joint_vel_limit, lam
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

	joint_vel_limit=np.radians([110,90,90,150,120,235]*2)

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

	###second arm base pose
	base_R=np.array([[-1,0,0],[0,-1,0],[0,0,1]])
	base_p=np.array([5000,0,0])

	###path constraints, position constraint and curve normal constraint
	cons = ({'type': 'eq', 'fun': lambda x:  fwd_all(x.reshape((-1,6*2))[:,:6]).p_all.flatten()-fwd_all(x.reshape((-1,6*2))[:,6:],base_R,base_p).p_all.flatten()-curve_cs_p.flatten()},
			{'type': 'eq', 'fun': lambda x:  np.dot(fwd_all(x.reshape((-1,6*2))[:,:6]).R_all, fwd_all(x.reshape((-1,6*2))[:,6:],base_R,base_p).R_all.T)[:,:,-1].flatten()-curve_cs_R[:,:,-1].flatten()})
	lowerer_limit=np.radians([-220.,-40.,-180.,-300.,-120.,-360.]*2)+0.001*np.ones(6*2)
	upper_limit=np.radians([220.,160.,70.,300.,120.,360.]*2)-0.001*np.ones(6*2)
	bnds=tuple(zip(lowerer_limit,upper_limit))*len(curve_js)

	res = minimize(opt_fun, curve_js, method='SLSQP',tol=1e-10,bounds=bnds,constraints=cons)
	###trust-constr not working
	print(res)
	print('originial: ',opt_fun(curve_js),'optimized: ',opt_fun(res.x.reshape((-1,6))))

	# print(res.x.reshape((-1,6*2)))
	dlam_out=calc_lamdot(res.x.reshape((-1,6*2)),lam,joint_vel_limit,1)


	plt.plot(lam[:-1],dlam_out,label="lambda_dot_max")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.title("max lambda_dot vs lambda (path index)")
	plt.savefig("velocity-constraint_js.png")
	plt.show()
	

if __name__ == "__main__":
	main()