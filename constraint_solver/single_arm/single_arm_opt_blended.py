import numpy as np
from pandas import *
import sys, traceback, time
from general_robotics_toolbox import *
import matplotlib.pyplot as plt

sys.path.append('../')
from constraint_solver import *

def main():
	data = read_csv("../../simulation/robotstudio_sim/scripts/fitting_output_new/threshold0.5/curve_fit.csv")
	curve_x=data['x'].tolist()
	curve_y=data['y'].tolist()
	curve_z=data['z'].tolist()
	curve_direction_x=data['R3'].tolist()
	curve_direction_y=data['R6'].tolist()
	curve_direction_z=data['R9'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_normal=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

	# col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	# data = read_csv("../../data/from_ge/Curve_in_base_frame2.csv", names=col_names)
	# curve_x=data['X'].tolist()
	# curve_y=data['Y'].tolist()
	# curve_z=data['Z'].tolist()
	# curve_direction_x=data['direction_x'].tolist()
	# curve_direction_y=data['direction_y'].tolist()
	# curve_direction_z=data['direction_z'].tolist()
	# curve=np.vstack((curve_x, curve_y, curve_z)).T
	# curve_normal=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

	curve=np.vstack((curve_x, curve_y, curve_z)).T

	###get breakpoints
	data = read_csv('../../simulation/robotstudio_sim/scripts/fitting_output_new/threshold0.5/command.csv')
	breakpoints=np.array(data['breakpoints'].tolist())
	primitives=data['primitives'].tolist()[1:]

	opt=lambda_opt(curve,curve_normal,robot1=abb6640(d=50),breakpoints=breakpoints,primitives=primitives)

	print(breakpoints)
	###path constraints, position constraint and curve normal constraint
	lowerer_limit=[-np.pi]
	upper_limit=[np.pi]
	bnds=tuple(zip(lowerer_limit,upper_limit))*len(opt.curve)
	bnds=tuple(zip([0],[2]))+bnds

	###diff evolution
	res = differential_evolution(opt.single_arm_global_opt_blended, bnds,workers=-1,
									x0 = np.zeros(1+len(opt.curve)),
									strategy='best1bin', maxiter=500,
									popsize=15, tol=1e-2,
									mutation=(0.5, 1), recombination=0.7,
									seed=None, callback=None, disp=False,
									polish=True, init='latinhypercube',
									atol=0.)

	print(res)
	pose_choice=int(np.floor(res.x[0]))
	theta=res.x[1:]

	for i in range(len(opt.curve)):
		
		if i==0:
			R_temp=direction2R(opt.curve_normal[0],-opt.curve_original[1]+opt.curve[0])
			R=np.dot(R_temp,Rz(theta[i]))
			q_out=[opt.robot1.inv(opt.curve[i],R)[pose_choice]]

		else:
			R_temp=direction2R(opt.curve_normal[i],-opt.curve[i]+opt.curve_original[opt.act_breakpoints[i]-1])
			R=np.dot(R_temp,Rz(theta[i]))
			###get closet config to previous one
			q_inv_all=opt.robot1.inv(opt.curve[i],R)
			temp_q=q_inv_all-q_out[-1]
			order=np.argsort(np.linalg.norm(temp_q,axis=1))
			q_out.append(q_inv_all[order[0]])

	


	q_out=np.array(q_out)
	####output to trajectory csv
	df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
	df.to_csv('trajectory/all_theta_opt_blended/arm1.csv',header=False,index=False)


	# curve_blend_js,dqdlam_list,spl_list,merged_idx=blend_js2(q_out,opt.breakpoints,opt.lam_original)
	# dlam_max1,dlam_max2=est_lamdot(dqdlam_list,opt.breakpoints,opt.lam_original,spl_list,merged_idx,opt.robot1)
	# lam_movej_seg_idx=(opt.lam_original[breakpoints[:-1]]+opt.lam_original[breakpoints[1:]])/2
	# lam_est=np.hstack((lam_movej_seg_idx,opt.lam_original[breakpoints[1:-1]]))
	# lam_est, lamdot_est = zip(*sorted(zip(lam_est, np.hstack((dlam_max1,dlam_max2)))))

	lam_blended,q_blended=blend_js_from_primitive(q_out,opt.curve_original,opt.breakpoints,opt.lam_original,opt.primitives,opt.robot1)
	dlam=calc_lamdot(q_blended,lam_blended,opt.robot1,1)

	###############################################restore 50,000 points with primitives#############################################
	theta_all=[]
	for i in range(len(theta)-1):
		theta_all=np.append(theta_all,np.linspace(theta[i],theta[i+1],(opt.act_breakpoints[i+1]-opt.act_breakpoints[i])))
	theta_all=np.append(theta_all,theta[-1]*np.ones(len(curve)-len(theta_all)))

	for i in range(len(theta_all)):
		if i==0:
			R_temp=direction2R(curve_normal[i],-curve[i+1]+curve[i])
			R=np.dot(R_temp,Rz(theta_all[i]))
			q_out=[opt.robot1.inv(curve[i],R)[pose_choice]]

		else:
			R_temp=direction2R(curve_normal[i],-curve[i]+curve[i-1])
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

	plt.plot(lam_blended[1:],dlam,label="lambda_dot_max")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.ylim([500,2000])
	plt.title("max lambda_dot vs lambda (path index)")
	plt.savefig("trajectory/all_theta_opt_blended/results.png")
	plt.show()

	


if __name__ == "__main__":
	main()