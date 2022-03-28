import numpy as np
from pandas import *
import sys, traceback, time
from general_robotics_toolbox import *
import matplotlib.pyplot as plt

sys.path.append('../')
from constraint_solver import *

def main():
	data = read_csv("../../simulation/robotstudio_sim/scripts/fitting_output_new/threshold0.1/curve_fit.csv")
	curve_x=data['x'].tolist()
	curve_y=data['y'].tolist()
	curve_z=data['z'].tolist()
	curve_direction_x=data['R3'].tolist()
	curve_direction_y=data['R6'].tolist()
	curve_direction_z=data['R9'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_normal=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

	###get breakpoints
	data = read_csv('../../simulation/robotstudio_sim/scripts/fitting_output_new/threshold0.1/command.csv')
	breakpoints=np.array(data['breakpoints'].tolist())
	breakpoints[1:]=breakpoints[1:]-1


	opt=lambda_opt(curve,curve_normal,robot1=abb6640(d=50),idx=breakpoints)

	###path constraints, position constraint and curve normal constraint
	lowerer_limit=[-np.pi]
	upper_limit=[np.pi]
	bnds=tuple(zip(lowerer_limit,upper_limit))*len(opt.curve)


	###diff evolution
	res = differential_evolution(opt.single_arm_global_opt, bnds,workers=-1,
									x0 = np.zeros(len(opt.curve)),
									strategy='best1bin', maxiter=500,
									popsize=15, tol=1e-2,
									mutation=(0.5, 1), recombination=0.7,
									seed=None, callback=None, disp=False,
									polish=True, init='latinhypercube',
									atol=0.)

	print(res)

	for i in range(len(opt.curve)):
		if i==0:
			R_temp=opt.direction2R(opt.curve_normal[0],-opt.curve_original[1]+opt.curve[0])
			R=np.dot(R_temp,Rz(res.x[i]))
			q_out=[opt.robot1.inv(opt.curve[i],R)[0]]

		else:
			R_temp=opt.direction2R(opt.curve_normal[i],-opt.curve[i]+opt.curve_original[opt.idx[i]-1])
			R=np.dot(R_temp,Rz(res.x[i]))
			###get closet config to previous one
			q_inv_all=opt.robot1.inv(opt.curve[i],R)
			temp_q=q_inv_all-q_out[-1]
			order=np.argsort(np.linalg.norm(temp_q,axis=1))
			q_out.append(q_inv_all[order[0]])

	


	q_out=np.array(q_out)
	####output to trajectory csv
	df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
	df.to_csv('trajectory/all_theta_opt_blended/arm1.csv',header=False,index=False)


	dlam_out=calc_lamdot(q_out,opt.lam[:len(q_out)],opt.robot1,1)

	###############################################restore 50,000 points#############################################
	theta_all=[]
	for i in range(len(res.x)-1):
		theta_all=np.append(theta_all,np.linspace(res.x[i],res.x[i+1],(opt.idx[i+1]-opt.idx[i])))
	theta_all=np.append(theta_all,res.x[-1]*np.ones(len(curve)-len(theta_all)))

	for i in range(len(theta_all)):
		if i==0:
			R_temp=opt.direction2R(curve_normal[i],-curve[i+1]+curve[i])
			R=np.dot(R_temp,Rz(theta_all[i]))
			q_out=[opt.robot1.inv(curve[i],R)[0]]

		else:
			R_temp=opt.direction2R(curve_normal[i],-curve[i]+curve[i-1])
			R=np.dot(R_temp,Rz(theta_all[i]))
			###get closet config to previous one
			q_inv_all=opt.robot1.inv(curve[i],R)
			temp_q=q_inv_all-q_out[-1]
			order=np.argsort(np.linalg.norm(temp_q,axis=1))
			q_out.append(q_inv_all[order[0]])

	q_out=np.array(q_out)

	###output to csv
	df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
	df.to_csv('trajectory/all_theta_opt_blended/all_theta_opt_js.csv',header=False,index=False)
	####################################################################################################################

	plt.plot(opt.lam[1:len(q_out)-1],dlam_out,label="lambda_dot_max")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.ylim([500,2000])
	plt.title("max lambda_dot vs lambda (path index)")
	plt.savefig("trajectory/all_theta_opt_blended/results.png")
	plt.show()

	


if __name__ == "__main__":
	main()