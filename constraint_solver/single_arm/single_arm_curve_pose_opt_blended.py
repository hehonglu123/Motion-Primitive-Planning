import sys, yaml
sys.path.append('../')
from constraint_solver import *

def main():
	###read actual curve
	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = data = read_csv("../../data/from_ge/relative_path.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_normal=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

	###get breakpoints
	data = read_csv('../../simulation/robotstudio_sim/scripts/fitting_output_new/threshold0.5/command.csv')
	breakpoints=np.array(data['breakpoints'].tolist())
	primitives=data['primitives'].tolist()[1:]

	robot=abb6640(d=50)
	opt=lambda_opt(curve,curve_normal,robot1=robot,breakpoints=breakpoints,primitives=primitives)


	###path constraints, position constraint and curve normal constraint
	lowerer_limit=[-np.pi]
	upper_limit=[np.pi]
	bnds=tuple(zip(lowerer_limit,upper_limit))*len(opt.curve)

	pose_lower_limit=np.array([-5,-5,-5,-3000,-3000,-3000])
	pose_upper_limit=-pose_lower_limit
	bnds=tuple(zip([0],[8]))+tuple(zip(pose_lower_limit,pose_upper_limit))+bnds
	###x: [pose_choice,blade k*theta, blade position, theta@breakpoints]

	res = differential_evolution(opt.curve_pose_opt_blended, bnds, args=None,workers=-1,
									x0 = [0,1.20919957785,1.20919957785,1.20919957785,2700,-800,500]+[0]*len(opt.curve),
									strategy='best1bin', maxiter=2000,
									popsize=15, tol=1e-10,
									mutation=(0.5, 1), recombination=0.7,
									seed=None, callback=None, disp=False,
									polish=True, init='latinhypercube',
									atol=0.)

	print(res)
	pose_choice=int(np.floor(res.x[0]))
	blade_theta=np.linalg.norm(res.x[1:4])	###pose rotation angle
	k=res.x[1:4]/blade_theta					###pose rotation axis
	shift=res.x[4:7]					###pose translation
	theta=res.x[7:]					###remaining DOF @breakpoints

	R_curve=rot(k,blade_theta)
	curve_new=np.dot(R_curve,opt.curve.T).T+np.tile(shift,(len(opt.curve),1))
	curve_normal_new=np.dot(R_curve,opt.curve_normal.T).T
	curve_originial_new=np.dot(R_curve,opt.curve_original.T).T+np.tile(shift,(len(opt.curve_original),1))
	curve_pose=np.vstack((np.hstack((R_curve,np.array([shift/1000.]).T)),np.array([0,0,0,1])))

	for i in range(len(opt.curve)):
		if i==0:
			R_temp=direction2R(curve_normal_new[0],-curve_originial_new[1]+curve_new[0])

			R=np.dot(R_temp,Rz(theta[i]))
			try:
				q_out=[opt.robot1.inv(curve_new[i],R)[pose_choice]]
			except:
				traceback.print_exc()

		else:
			R_temp=direction2R(curve_normal_new[i],-curve_new[i]+curve_originial_new[opt.act_breakpoints[i]-1])

			R=np.dot(R_temp,Rz(theta[i]))
			try:
				###get closet config to previous one
				q_inv_all=opt.robot1.inv(curve_new[i],R)
				temp_q=q_inv_all-q_out[-1]
				order=np.argsort(np.linalg.norm(temp_q,axis=1))
				q_out.append(q_inv_all[order[0]])
			except:
				traceback.print_exc()

	q_out=np.array(q_out)
	####output to trajectory csv
	df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
	df.to_csv('trajectory/curve_pose_opt_blended/arm1.csv',header=False,index=False)
	with open(r'trajectory/curve_pose_opt_blended/curve_pose.yaml', 'w') as file:
		documents = yaml.dump({'H':curve_pose.tolist()}, file)

	lam_blended,q_blended=blend_js_from_primitive(q_out,curve_originial_new,opt.breakpoints,opt.lam_original,opt.primitives,opt.robot1)
	dlam=calc_lamdot(q_blended,lam_blended,opt.robot1,1)

	plt.plot(lam_blended,dlam,label="lambda_dot_max")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.ylim([0,2000])
	plt.title("max lambda_dot vs lambda (path index)")
	plt.savefig("trajectory/curve_pose_opt_blended/results.png")
	plt.show()
if __name__ == "__main__":
	main()