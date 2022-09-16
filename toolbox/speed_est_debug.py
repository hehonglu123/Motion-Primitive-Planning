###speed estimation dual given second robot cmd speed
from MotionSend import MotionSend
from lambda_calc import *
from dual_arm import *

dataset='wood/'
data_dir="../data/"+dataset
solution_dir=data_dir+'dual_arm/'+'diffevo_pose1/'
cmd_dir=solution_dir+'50L/'

relative_path,robot1,robot2,base2_R,base2_p,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,data_dir,solution_dir,cmd_dir)

with open(data_dir+'dual_arm/tcp.yaml') as file:
    H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
robot1=abb6640(d=50, acc_dict_path='robot_info/6640acc.pickle')
robot2=abb1200(R_tool=H_tcp[:3,:3],p_tool=H_tcp[:-1,-1], acc_dict_path='robot_info/1200acc.pickle')


ms = MotionSend(robot2=robot2,base2_R=base2_R,base2_p=base2_p)

# exe_dir='../simulation/robotstudio_sim/multimove/'+dataset+solution_dir


v_cmds=[1000]
for v_cmd in v_cmds:

	###read actual exe file
	df=read_csv('../simulation/robotstudio_sim/multimove/wood/diffevo_pose1/curve_exe_v1000_z100.csv')
	# df=read_csv('../ILC/max_gradient/curve1_diffevo1/dual_iteration_2.csv')
	lam_exe, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R = ms.logged_data_analysis_multimove(df,base2_R,base2_p,realrobot=True)

	lam_exe, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
		ms.chop_extension_dual(lam_exe, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])

	error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])
	print('max error: ', np.max(error))

	speed1=get_speed(curve_exe1,timestamp)
	speed2=get_speed(curve_exe2,timestamp)

	lam2=calc_lam_cs(curve_exe2)
	lam=lam_exe

	speed_est,speed1_est,speed2_est=traj_speed_est_dual(robot1,robot2,curve_exe_js1,curve_exe_js2,base2_R,base2_p,lam,v_cmd)	


	plt.plot(lam_exe,speed_est,label='estimated')
	plt.plot(lam_exe,speed,label='actual')


	plt.legend()
	plt.xlabel('lambda (mm)')
	plt.ylabel('Speed (mm/s)')
	plt.title('Speed Estimation for v'+str(v_cmd))
	plt.show()

	plt.plot(lam_exe,speed1_est,label='estimated')
	plt.plot(lam_exe,speed1,label='actual')


	plt.legend()
	plt.xlabel('lambda (mm)')
	plt.ylabel('Speed (mm/s)')
	plt.title('TCP1 Speed Estimation for v'+str(v_cmd))
	plt.show()

	plt.plot(lam_exe,speed2_est,label='estimated')
	plt.plot(lam_exe,speed2,label='actual')


	plt.legend()
	plt.xlabel('lambda (mm)')
	plt.ylabel('Speed (mm/s)')
	plt.title('TCP2 Speed Estimation for v'+str(v_cmd))
	plt.show()
