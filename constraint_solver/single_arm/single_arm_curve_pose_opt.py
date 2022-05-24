import sys, yaml
sys.path.append('../')
from constraint_solver import *

def main():
	data_dir='../../data/wood/'
	###read actual curve
	curve_dense = read_csv(data_dir+"Curve_dense.csv",header=None).values


	robot=abb6640(d=50)
	opt=lambda_opt(curve_dense[:,:3],curve_dense[:,3:],robot1=robot,steps=500)

	#read in initial curve pose
	with open(data_dir+'blade_pose.yaml') as file:
		curve_pose = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

	k,theta=R2rot(curve_pose[:3,:3])

	###path constraints, position constraint and curve normal constraint
	lowerer_limit=np.array([-2*np.pi,-2*np.pi,-2*np.pi,0,-3000,0,-np.pi])
	upper_limit=np.array([2*np.pi,2*np.pi,2*np.pi,3000,3000,3000,np.pi])
	bnds=tuple(zip(lowerer_limit,upper_limit))


	res = differential_evolution(opt.curve_pose_opt, bnds, args=None,workers=-1,
									x0 = np.hstack((k*theta,curve_pose[:-1,-1],[0])),
									strategy='best1bin', maxiter=200,
									popsize=15, tol=1e-10,
									mutation=(0.5, 1), recombination=0.7,
									seed=None, callback=None, disp=False,
									polish=True, init='latinhypercube',
									atol=0.)
	


	print(res)
	theta0=np.linalg.norm(res.x[:3])
	k=res.x[:3]/theta0
	shift=res.x[3:-1]
	theta1=res.x[-1]

	R_curve=rot(k,theta0)
	curve_new=np.dot(R_curve,opt.curve.T).T+np.tile(shift,(len(opt.curve),1))
	curve_normal_new=np.dot(R_curve,opt.curve_normal.T).T

	curve_pose=np.vstack((np.hstack((R_curve,np.array([shift/1000.]).T)),np.array([0,0,0,1])))


	R_temp=direction2R(curve_normal_new[0],-curve_new[1]+curve_new[0])
	R=np.dot(R_temp,Rz(theta1))
	q_init=robot.inv(curve_new[0],R)[0]


	# q_out=opt.single_arm_stepwise_optimize(q_init,curve_new,curve_normal_new)
	q_out=opt.followx(curve_new,curve_normal_new)

	####output to trajectory csv
	df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
	df.to_csv('trajectory/curve_pose_opt/arm1.csv',header=False,index=False)
	with open(r'trajectory/curve_pose_opt/curve_pose.yaml', 'w') as file:
		documents = yaml.dump({'H':curve_pose.tolist()}, file)

	dlam_out=calc_lamdot(q_out,opt.lam,opt.robot1,1)


	plt.plot(opt.lam,dlam_out,label="lambda_dot_max")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	# plt.ylim([1000,4000])
	plt.title("max lambda_dot vs lambda (path index)")
	plt.savefig("trajectory/curve_pose_opt/results.png")
	plt.show()
if __name__ == "__main__":
	main()