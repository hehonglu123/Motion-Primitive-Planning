import sys
sys.path.append('../')
from constraint_solver import *


def main():
	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	# data = read_csv("../../data/from_ge/Curve_in_base_frame2.csv", names=col_names)
	data = read_csv("../../data/wood/Curve_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_normal=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

	robot=abb6640(d=50)
	opt=lambda_opt(curve,curve_normal,robot1=robot,steps=500)
	# q_init=[0.627591343,	0.839862344,	-0.238013642,	1.679129375,	-0.901227684,	0.79092621]
	# q_init=[-5.39E-10,	0.42424468,	0.863930807,	0,	-1.811774263,	0]
	q_init=[-1.71E-06,	0.458127951,	0.800479092,	-5.65E-06,	-1.782205819,	-7.34E-06]

	q_out=opt.single_arm_stepwise_optimize(q_init)

	####output to trajectory csv
	df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
	df.to_csv('trajectory/stepwise_opt/arm1.csv',header=False,index=False)

	dlam_out=calc_lamdot(q_out,opt.lam,opt.robot1,1)


	plt.plot(opt.lam,dlam_out,label="lambda_dot_max")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.ylim([0,2000])
	plt.title("max lambda_dot vs lambda (path index)")
	plt.savefig("trajectory/stepwise_opt/results.png")
	plt.show()
	

if __name__ == "__main__":
	main()