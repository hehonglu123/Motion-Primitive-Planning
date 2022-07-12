import sys
sys.path.append('../')
sys.path.append('../../toolbox')
from constraint_solver import *
from error_check import *

def main():

	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	# train_data = read_csv("../../train_data/from_ge/Curve_in_base_frame2.csv", names=col_names)
	# train_data = read_csv("../../train_data/wood/Curve_in_base_frame.csv", names=col_names)
	curve = read_csv("../../train_data/from_NX/Curve_in_base_frame.csv", names=col_names).values
	curve_js=read_csv("../../train_data/from_NX/Curve_js.csv", names=col_names).values

	robot=abb6640(d=50)
	opt=lambda_opt(curve[:,:3],curve[:,3:],robot1=robot,steps=50000)

	lamdot_des=500
	q_all,lam_out,curve_out,curve_normal_out,act_speed=opt.single_arm_stepwise_optimize2(curve_js[0],lamdot_des)

	###plot results
	plt.figure(0)
	ax = plt.axes(projection='3d')
	ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='red')
	ax.plot3D(curve_out[:,0], curve_out[:,1], curve_out[:,2], c='green')
	# plt.show()

	print(calc_max_error_w_normal(curve_out[2:],curve[:,:3],curve_normal_out[2:],curve[:,3:]))
	# df=DataFrame({'j1':q_all[:,0],'j2':q_all[:,1],'j3':q_all[:,2],'j4':q_all[:,3],'j5':q_all[:,4],'j6':q_all[:,5]})
	# df.to_csv('curve_qp_js.csv',header=False,index=False)

	lamdot=calc_lamdot(q_all,lam_out,robot,1)

	plt.figure(1)
	plt.plot(lam_out[1:],act_speed)
	plt.title("TCP speed (jacobian) vs lambda, v_des= "+str(lamdot_des))
	plt.ylabel('speed (mm/s)')
	plt.xlabel('lambda (mm)')


	plt.figure(2)
	plt.plot(lam_out,lamdot, label='Lambda Dot')
	plt.title("lamdot vs lambda, v_des= "+str(lamdot_des))
	plt.ylabel('lamdot (mm/s)')
	plt.xlabel('lambda (mm)')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()