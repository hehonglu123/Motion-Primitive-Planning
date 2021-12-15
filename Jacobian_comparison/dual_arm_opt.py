import numpy as np
from pandas import *
import sys, traceback, time
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qpsolvers import solve_qp

sys.path.append('../toolbox')
from robot_def import *
from lambda_calc import *

def stepwise_optimize(q_init1,q_init2,curve,curve_normal,base2_R,base2_p):
	#curve_normal: expressed in second robot tool frame

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

				pose2_base_now=fwd(q_all2[-1],base2_R,base2_p)

				print(pose1_now.p-pose2_base_now.p-curve[i], np.linalg.norm(np.dot(pose2_base_now.R.T,pose1_now.R[:,-1])-curve_normal[i]))
				error_fb=np.linalg.norm(pose1_now.p-pose2_base_now.p-curve[i])+np.linalg.norm(np.dot(pose2_base_now.R.T,pose1_now.R[:,-1])-curve_normal[i])
				
				########################################################first robot###########################################
				w=0.2
				Kq=.01*np.eye(6)    #small value to make sure positive definite
				KR=np.eye(3)        #gains for position and orientation error

				J1=jacobian(q_all1[-1])        #calculate current Jacobian
				J1p=J1[3:,:]
				J1R=J1[:3,:] 
				H=np.dot(np.transpose(J1p),J1p)+Kq+w*np.dot(np.transpose(J1R),J1R)
				H=(H+np.transpose(H))/2

				vd=curve[i]-(pose1_now.p-pose2_base_now.p)
				k=np.cross(pose1_now.R[:,-1],np.dot(pose2_base_now.R,curve_normal[i]))

				k=k/np.linalg.norm(k)
				theta=-np.arctan2(np.linalg.norm(np.cross(pose1_now.R[:,-1],np.dot(pose2_base_now.R,curve_normal[i]))), np.dot(pose1_now.R[:,-1],np.dot(pose2_base_now.R,curve_normal[i])))

				k=np.array(k)
				s=np.sin(theta/2)*k         #eR2
				wd=-np.dot(KR,s)
				f=-np.dot(np.transpose(J1p),vd)-w*np.dot(np.transpose(J1R),wd)
				qdot=solve_qp(H,f)

				# q_all1.append(q_all1[-1]+qdot)

				########################################################first robot###########################################
				J2=jacobian(q_all2[-1])        #calculate current Jacobian, mapped to robot1 base frame
				J2p=np.dot(base2_R,J2[3:,:])
				J2R=np.dot(base2_R,J2[:3,:] )
				H=np.dot(np.transpose(J2p),J2p)+Kq+w*np.dot(np.transpose(J2R),J2R)
				H=(H+np.transpose(H))/2

				vd=-vd
				k=np.cross(pose2_base_now.R[:,-1],np.dot(pose2_base_now.R,-curve_normal[i]))

				k=k/np.linalg.norm(k)
				theta=-np.arctan2(np.linalg.norm(np.cross(pose2_base_now.R[:,-1],np.dot(pose2_base_now.R,-curve_normal[i]))), np.dot(pose2_base_now.R[:,-1],np.dot(pose2_base_now.R,-curve_normal[i])))

				k=np.array(k)
				s=np.sin(theta/2)*k         #eR2
				wd=-np.dot(KR,s)
				f=-np.dot(np.transpose(J2p),vd)-w*np.dot(np.transpose(J2R),wd)
				qdot=solve_qp(H,f)

				q_all2.append(q_all2[-1]+qdot)
		except:
			q_out1.append(q_all1[-1])
			q_out2.append(q_all2[-1])
			traceback.print_exc()
			break

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

	###decrease curve density to simplify computation
	step=1000	
	relative_path=relative_path[0:-1:step]
	relative_path_direction=relative_path_direction[0:-1:step]

	###convert relative curve normal to robot2 tool frame
	R2_tool=np.array([	[0,0,1],
						[0,1,0],
						[-1,0,0]])
	R2_convert=np.dot(base2_R,R2_tool)
	relative_path_direction_new=np.dot(R2_convert.T,relative_path_direction.T).T


	curve_cs_p=[]
	curve_cs_R=[]
	###find path length
	lam=[0]
	for i in range(len(relative_path)-1):
		lam.append(lam[-1]+np.linalg.norm(relative_path[i+1]-relative_path[i]))
	###normalize lam
	lam=np.array(lam)/lam[-1]

	###stepwise qp solver
	q_out1, q_out2=stepwise_optimize(q_init1,q_init2,relative_path,relative_path_direction_new,base2_R,base2_p)


	print(q_out1)
	dlam_out=calc_lamdot(q_out1,lam[:len(q_out1)],joint_vel_limit[:6],1)



	plt.plot(lam[:-1],dlam_out,label="lambda_dot_max")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.title("max lambda_dot vs lambda (path index)")
	plt.savefig("velocity-constraint_js.png")
	plt.show()
	

if __name__ == "__main__":
	main()