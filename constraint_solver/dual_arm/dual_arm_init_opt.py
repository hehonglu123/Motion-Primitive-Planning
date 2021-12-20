import numpy as np
from pandas import *
import sys, traceback, time
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, shgo, NonlinearConstraint, minimize
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

def opt_fun(x,lam,joint_vel_limit,curve,curve_normal,base2_R,base2_p,upper_limit,lowerer_limit):
	q_init2=x[:-1]

	pose2_world_now=fwd(q_init2,base2_R,base2_p)

	R_temp=direction2R(np.dot(pose2_world_now.R,curve_normal[0]),-curve[1]+curve[0])
	R=np.dot(R_temp,Rz(x[-1]))
	try:
		q_init1=inv(pose2_world_now.p,R)[0]
		q_out1,q_out2=stepwise_optimize(q_init1,q_init2,curve,curve_normal,base2_R,base2_p,upper_limit,lowerer_limit)
	except:
		# traceback.print_exc()
		return 999

	
	dlam=calc_lamdot(np.hstack((q_out1,q_out2)),lam[:len(q_out2)],joint_vel_limit,1)
	print(min(dlam))
	return -min(dlam)


def stepwise_optimize(q_init1,q_init2,curve,curve_normal,base2_R,base2_p,upper_limit,lowerer_limit):

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

				pose1_now=fwd(q_all1[-1])
				pose2_now=fwd(q_all2[-1])

				pose2_world_now=fwd(q_all2[-1],base2_R,base2_p)

				# print(pose1_now.p-pose2_world_now.p-curve[i], np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])-curve_normal[i]))
				error_fb=np.linalg.norm(pose1_now.p-pose2_world_now.p-curve[i])+np.linalg.norm(np.dot(pose2_world_now.R.T,pose1_now.R[:,-1])-curve_normal[i])
				
				########################################################first robot###########################################
				w=0.2
				Kq=.01*np.eye(6)    #small value to make sure positive definite
				KR=np.eye(3)        #gains for position and orientation error
				Kp=0.1				#gains for vd

				J1=jacobian(q_all1[-1])        #calculate current Jacobian
				J1p=J1[3:,:]
				J1R=J1[:3,:] 
				H=np.dot(np.transpose(J1p),J1p)+Kq+w*np.dot(np.transpose(J1R),J1R)
				H=(H+np.transpose(H))/2

				vd=Kp*(curve[i]-(pose1_now.p-pose2_world_now.p))
				k=np.cross(pose1_now.R[:,-1],np.dot(pose2_world_now.R,curve_normal[i]))

				k=k/np.linalg.norm(k)
				theta=-np.arctan2(np.linalg.norm(np.cross(pose1_now.R[:,-1],np.dot(pose2_world_now.R,curve_normal[i]))), np.dot(pose1_now.R[:,-1],np.dot(pose2_world_now.R,curve_normal[i])))

				k=np.array(k)
				s=np.sin(theta/2)*k         #eR2
				wd=-np.dot(KR,s)
				f=-np.dot(np.transpose(J1p),vd)-w*np.dot(np.transpose(J1R),wd)
				qdot=solve_qp(H,f)

				###TODO:
				#cap to joint limits
				q_all1.append(q_all1[-1]+qdot)

				########################################################Second robot###########################################
				J2=jacobian(q_all2[-1])        #calculate current Jacobian, mapped to robot1 base frame
				J2p=np.dot(base2_R,J2[3:,:])
				J2R=np.dot(base2_R,J2[:3,:] )
				H=np.dot(np.transpose(J2p),J2p)+Kq+w*np.dot(np.transpose(J2R),J2R)
				H=(H+np.transpose(H))/2

				vd=-vd
				# k=np.cross(pose2_world_now.R[:,-1],np.dot(pose2_world_now.R,-curve_normal[i]))

				# k=k/np.linalg.norm(k)
				# theta=-np.arctan2(np.linalg.norm(np.cross(pose2_world_now.R[:,-1],np.dot(pose2_world_now.R,-curve_normal[i]))), np.dot(pose2_world_now.R[:,-1],np.dot(pose2_world_now.R,-curve_normal[i])))
				theta=-theta

				k=np.array(k)
				s=np.sin(theta/2)*k         #eR2
				wd=-np.dot(KR,s)
				f=-np.dot(np.transpose(J2p),vd)-w*np.dot(np.transpose(J2R),wd)
				qdot=solve_qp(H,f)
				###TODO:
				#cap to joint limits
				q_all2.append(q_all2[-1]+qdot)
		except:
			q_out1.append(q_all1[-1])
			q_out2.append(q_all2[-1])
			# traceback.print_exc()
			raise AssertionError
			return

		q_out1.append(q_all1[-1])
		q_out2.append(q_all2[-1])

	q_out1=np.array(q_out1)
	q_out2=np.array(q_out2)
	return q_out1, q_out2


def main():
	###read actual curve
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("curve_poses/dual_arm/arm1_js.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js1=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T
	q_init1=curve_js1[0]

	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("curve_poses/dual_arm/arm2_js.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js2=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T
	q_init2=curve_js2[0]

	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = read_csv("curve_poses/dual_arm/relative_path.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()
	relative_path=np.vstack((curve_x, curve_y, curve_z)).T
	relative_path_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

	###second arm base pose
	base2_R=np.array([[-1,0,0],[0,-1,0],[0,0,1]])
	base2_p=np.array([6000,0,0])

	joint_vel_limit=np.radians([110,90,90,150,120,235]*2)
	joint_upper_limit=np.radians([220.,160.,70.,300.,120.,360.])
	joint_lowerer_limit=np.radians([-220.,-40.,-180.,-300.,-120.,-360.])

	###decrease curve density to simplify computation
	step=1000	
	relative_path=relative_path[0:-1:step]
	relative_path_direction=relative_path_direction[0:-1:step]

	###find path length
	lam=[0]
	for i in range(len(relative_path)-1):
		lam.append(lam[-1]+np.linalg.norm(relative_path[i+1]-relative_path[i]))
	###normalize lam
	lam=np.array(lam)/lam[-1]

	###convert relative curve normal to robot2 tool frame
	R2_tool=np.array([	[0,0,1],
						[0,1,0],
						[-1,0,0]])
	R2_convert=np.dot(base2_R,R2_tool)
	relative_path_direction_new=np.dot(R2_convert.T,relative_path_direction.T).T


	###path constraints, position constraint and curve normal constraint
	lowerer_limit=np.append(joint_lowerer_limit,[-np.pi])
	upper_limit=np.append(joint_upper_limit,[np.pi])
	bnds=tuple(zip(lowerer_limit,upper_limit))

	# ###opt variable: x=[rob2joint,theta(eef angle of first robot)]
	# res = minimize(opt_fun, np.append(q_init2,[0]), args=(lam,joint_vel_limit,relative_path,relative_path_direction_new,base2_R,base2_p,upper_limit,lowerer_limit,), method='SLSQP',tol=1e-10,bounds=bnds)

	res = differential_evolution(opt_fun, bnds, args=(lam,joint_vel_limit,relative_path,relative_path_direction_new,base2_R,base2_p,upper_limit,lowerer_limit,),workers=-1,
									x0 = np.append(q_init2,[0]),
									strategy='best1bin', maxiter=200,
									popsize=15, tol=1e-10,
									mutation=(0.5, 1), recombination=0.7,
									seed=None, callback=None, disp=False,
									polish=True, init='latinhypercube',
									atol=0.)
	


	print(res)

	q_init2=res.x[:-1]
	pose2_world_now=fwd(q_init2,base2_R,base2_p)

	R_temp=direction2R(np.dot(pose2_world_now.R,relative_path_direction_new[0]),-relative_path[1]+relative_path[0])
	R=np.dot(R_temp,Rz(res.x[-1]))

	q_init1=inv(pose2_world_now.p,R)[0]
	q_out1,q_out2=stepwise_optimize(q_init1,q_init2,relative_path,relative_path_direction_new,base2_R,base2_p,upper_limit,lowerer_limit)

	###stepwise qp solver
	q_out1, q_out2=stepwise_optimize(q_init1,q_init2,relative_path,relative_path_direction_new,base2_R,base2_p,upper_limit,lowerer_limit)
	
	dlam=calc_lamdot(np.hstack((q_out1,q_out2)),lam[:len(q_out2)],joint_vel_limit,1)

	plt.plot(lam[:len(q_out2)-1],dlam,label="lambda_dot_max")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.title("DUALARM max lambda_dot vs lambda (path index)")
	plt.ylim([0.5,3.5])
	plt.savefig("velocity-constraint_js.png")
	plt.show()
	

if __name__ == "__main__":
	main()