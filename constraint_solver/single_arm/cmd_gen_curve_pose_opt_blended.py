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
	primitives=data['primitives'].tolist()
	points=data['points'].tolist()

	points_list=[]
	for i in range(len(breakpoints)):
		if primitives[i]=='movel_fit':
			point=extract_points(primitives[i],points[i])
			points_list.append(point)
		elif primitives[i]=='movec_fit':
			point1,point2=extract_points(primitives[i],points[i])
			points_list.append([point1,point2])
		else:
			point=extract_points(primitives[i],points[i])
			points_list.append(point)

	robot=abb6640(d=50)
	opt=lambda_opt(curve,curve_normal,robot1=robot,breakpoints=breakpoints,primitives=primitives[1:])

	# x=np.array([ 2.18654984e-01,  1.55026572e+00,  1.08374149e+00,  1.07663488e+00,
	# 	2.92766347e+03, -1.71953011e+03, -2.76534209e+01, -1.86046911e+00,
	#    -2.12585191e+00, -2.17532131e+00, -2.21957886e+00, -2.19957566e+00,
	#    -1.99789812e+00, -1.70873595e+00, -1.41831834e+00, -9.47779982e-01,
	#    -0.9])
	x=np.array([ 2.18654984e-01,  1.55026572e+00,  1.08374149e+00,  1.07663488e+00,
		2.92766347e+03, -1.71953011e+03, -2.76534209e+01, -1.86046911e+00,
	   -2.12585191e+00, -2.17532131e+00, -2.21957886e+00, -2.19957566e+00,
	   -1.99789812e+00, -1.70873595e+00, -1.41831834e+00, -9.47779982e-01,
	   -6.34510532e-01])

	pose_choice=int(np.floor(x[0]))
	blade_theta=np.linalg.norm(x[1:4])	###pose rotation angle
	k=x[1:4]/blade_theta					###pose rotation axis
	shift=x[4:7]					###pose translation
	theta=x[7:]					###remaining DOF @breakpoints

	R_curve=rot(k,blade_theta)
	curve_new=np.dot(R_curve,opt.curve.T).T+np.tile(shift,(len(opt.curve),1))
	curve_normal_new=np.dot(R_curve,opt.curve_normal.T).T
	curve_originial_new=np.dot(R_curve,opt.curve_original.T).T+np.tile(shift,(len(opt.curve_original),1))
	curve_pose=np.vstack((np.hstack((R_curve,np.array([shift/1000.]).T)),np.array([0,0,0,1])))

	##########################################################inv kin##################################
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


	###################################convert original points to new points based on new curve pose##################################
	R_original=np.array([[0,0,1],
						[1,0,0],
						[0,1,0]])
	shift_original=np.array([2700,-800,500])

	points_new=[]
	for i in range(len(breakpoints)):
		if primitives[i]=='movel_fit':
			point=np.dot(R_original.T,points_list[i]-shift_original)
			point_new=np.dot(R_curve,point)+shift
			points_new.append([point_new])
		elif primitives[i]=='movec_fit':
			point1=np.dot(R_original.T,points_list[i][0]-shift_original)
			point1_new=np.dot(R_curve,point1)+shift
			point2=np.dot(R_original.T,points_list[i][1]-shift_original)
			point2_new=np.dot(R_curve,point2)+shift
			points_new.append([point1_new,point2_new])
		else:
			point=q_out[breakpoints[i]]
			points_new.append([point])


	df=DataFrame({'breakpoints':data['breakpoints'].tolist(),'primitives':primitives,'points':points_new})
	df.to_csv('trajectory/curve_pose_opt_blended/command.csv',header=True,index=False)


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