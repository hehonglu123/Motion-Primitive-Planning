import sys, yaml
sys.path.append('../../toolbox')
from robots_def import *
from lambda_calc import *
# from tes_env import *
sys.path.append('../')
from constraint_solver import *

def main():
	dataset='curve_2'
	data_dir='../../data/'+dataset+'/'
	###read actual curve
	curve_dense = read_csv(data_dir+"Curve_dense.csv",header=None).values


	# robot=robot_obj('ABB_6640_180_255','../../config/ABB_6640_180_255_robot_default_config.yml',tool_file_path='../../config/paintgun.csv',d=50,acc_dict_path='../../toolbox/robot_info/6640acc_new.pickle')
	# robot = robot_obj('FANUC_m10ia','../../config/FANUC_m10ia_robot_default_config.yml',tool_file_path='../../config/paintgun.csv',d=50,acc_dict_path='../../config/FANUC_m10ia_acc_new.pickle')
	robot=robot_obj('MA2010_A0',def_path='../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun2.csv',\
    pulse2deg_file_path='../../config/MA2010_A0_pulse2deg_real.csv',d=50,acc_dict_path='../../config/acceleration/MA2010_A0.pickle')
    
	v_cmd=1000
	# opt=lambda_opt(curve_dense[:,:3],curve_dense[:,3:],robot1=robot,urdf_path='../../config/urdf/',curve_name=dataset,steps=500,v_cmd=v_cmd)
	opt=lambda_opt(curve_dense[:,:3],curve_dense[:,3:],robot1=robot,curve_name=dataset,steps=500,v_cmd=v_cmd)

	#read in initial curve pose
	# curve_pose=np.loadtxt(data_dir+'baseline/curve_pose.csv',delimiter=',')
	curve_pose=np.loadtxt(data_dir+'baseline_motoman/curve_pose.csv',delimiter=',')

	k,theta=R2rot(curve_pose[:3,:3])

	###path constraints, position constraint and curve normal constraint
	lowerer_limit=np.array([-2*np.pi,-2*np.pi,-2*np.pi,0,-3000,0,-np.pi])
	upper_limit=np.array([2*np.pi,2*np.pi,2*np.pi,3000,3000,3000,np.pi])
	bnds=tuple(zip(lowerer_limit,upper_limit))


	res = differential_evolution(opt.curve_pose_opt2, bnds, args=None,workers=-1,
									x0 = np.hstack((k*theta,curve_pose[:-1,-1],[0])),
									strategy='best1bin', maxiter=600,
									popsize=15, tol=1e-10,
									mutation=(0.5, 1), recombination=0.7,
									seed=None, callback=None, disp=True,
									polish=True, init='latinhypercube',
									atol=0.)
	
	# class Object(object):
	# 	pass
	# res=Object()
	# setattr(res, "x", np.array([ 5.22786277e+00,  4.30371672e+00,  3.25477272e+00,  1.25793114e+03, 7.65140344e+02,  6.87793008e+02, -2.82344434e-01]))


	print(res)
	theta0=np.linalg.norm(res.x[:3])
	k=res.x[:3]/theta0
	shift=res.x[3:-1]
	theta1=res.x[-1]

	R_curve=rot(k,theta0)
	curve_pose=np.vstack((np.hstack((R_curve,np.array([shift]).T)),np.array([0,0,0,1])))

	np.savetxt('trajectory/curve_pose.csv', curve_pose, delimiter=',')

	###get initial q
	curve_new=np.dot(R_curve,opt.curve.T).T+np.tile(shift,(len(opt.curve),1))
	curve_normal_new=np.dot(R_curve,opt.curve_normal.T).T

	R_temp=direction2R(curve_normal_new[0],-curve_new[1]+curve_new[0])
	R=np.dot(R_temp,Rz(theta1))
	q_init=robot.inv(curve_new[0],R)[0]

	#########################################restore only given points, saves time##########################################################
	q_out=opt.single_arm_stepwise_optimize(q_init,curve_new,curve_normal_new)
	# q_out=opt.followx(curve_new,curve_normal_new)

	####output to trajectory csv
	df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
	df.to_csv('trajectory/arm1.csv',header=False,index=False)
	df=DataFrame({'x':curve_new[:,0],'y':curve_new[:,1],'z':curve_new[:,2],'nx':curve_normal_new[:,0],'ny':curve_normal_new[:,1],'nz':curve_normal_new[:,2]})
	df.to_csv('trajectory/curve_pose_opt_cs.csv',header=False,index=False)
	#########################################restore only given points, END##########################################################

	# dlam_out=calc_lamdot(q_out,opt.lam,opt.robot1,1)
	speed=traj_speed_est(opt.robot1,q_out,opt.lam,opt.v_cmd)

	# plt.plot(opt.lam,dlam_out,label="lambda_dot_max")
	plt.plot(opt.lam,speed,label="speed est")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	# plt.ylim([1000,4000])
	plt.title("max lambda_dot vs lambda (path index)")
	plt.savefig("trajectory/results.png")

	###optional, solve for dense curve
	#########################################restore all 50,000 points, takes time##########################################################
	opt=lambda_opt(curve_dense[:,:3],curve_dense[:,3:],robot1=robot,steps=50000,v_cmd=v_cmd)
	curve_new=np.dot(R_curve,opt.curve.T).T+np.tile(shift,(len(opt.curve),1))
	curve_normal_new=np.dot(R_curve,opt.curve_normal.T).T

	q_out=opt.single_arm_stepwise_optimize(q_init,curve_new,curve_normal_new)
	####output to trajectory csv
	df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
	df.to_csv('trajectory/Curve_js.csv',header=False,index=False)
	df=DataFrame({'x':curve_new[:,0],'y':curve_new[:,1],'z':curve_new[:,2],'nx':curve_normal_new[:,0],'ny':curve_normal_new[:,1],'nz':curve_normal_new[:,2]})
	df.to_csv('trajectory/Curve_in_base_frame.csv',header=False,index=False)
	#########################################restore all 50,000 points, END##########################################################
	
	
	

	

if __name__ == "__main__":
	main()