###speed estimation dual given second robot cmd speed
from MotionSend import MotionSend
from lambda_calc import *
from dual_arm import *

dataset='wood/'
solution_dir='baseline/'
robot=abb6640(d=50, acc_dict_path='robot_info/6640acc.pickle')
curve = read_csv('../data/'+dataset+solution_dir+'/Curve_in_base_frame.csv',header=None).values

curve_js=read_csv('../data/'+dataset+solution_dir+'/Curve_js.csv',header=None).values
curve=curve[::1000]
curve_js=curve_js[::1000]
lam_original=calc_lam_cs(curve)

###get joint acceleration at each pose
joint_acc_limit=[]
for q in curve_js:
	joint_acc_limit.append(robot.get_acc(q))

ms = MotionSend()

breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd('../data/'+dataset+'baseline/100L/command.csv')
# breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd('../ILC/max_gradient/curve1_250_100L_multipeak/command.csv')
# breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd('../ILC/max_gradient/curve2_1100_100L_multipeak/command.csv')

# exe_dir='../simulation/robotstudio_sim/scripts/fitting_output/'+dataset+'/100L/'
exe_dir='../simulation/robotstudio_sim/scripts/recorded_data/'
v_cmds=[1000]
for v_cmd in v_cmds:

	###read actual exe file
	df = read_csv(exe_dir+"curve_exe"+"_v"+str(v_cmd)+"_z10.csv")
	lam_exe, curve_exe, curve_exe_R,curve_exe_js, act_speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)
	# lam_exe, curve_exe, curve_exe_R,curve_exe_js, act_speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, act_speed, timestamp,curve[0,:3],curve[-1,:3])

	v_cmd=2000
	speed_est=traj_speed_est(robot,curve_exe_js,lam_exe,v_cmd)


	qdot_all=np.gradient(curve_exe_js,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T
	qddot_all=np.gradient(qdot_all,axis=0)/np.tile([np.gradient(timestamp)],(6,1)).T
	qddot_violate_idx=np.array([])
	for i in range(len(curve_exe_js[0])):
		qddot_violate_idx=np.append(qddot_violate_idx,np.argwhere(np.abs(qddot_all[:,i])>robot.joint_acc_limit[i]))
	qddot_violate_idx=np.unique(qddot_violate_idx).astype(int)
	# for idx in qddot_violate_idx:
	# 	plt.axvline(x=lam_exe[idx],c='orange')
	

	plt.plot(lam_exe,speed_est,label='estimated')
	plt.plot(lam_exe[1:],act_speed,label='actual')

	plt.legend()
	plt.ylim([0,1.2*v_cmd])
	plt.xlabel('lambda (mm)')
	plt.ylabel('Speed (mm/s)')
	plt.title('Speed Estimation for v'+str(v_cmd))
	plt.show()

	plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='gray',label='original')
	ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c='red',label='execution')
	ax.scatter3D(curve_exe[qddot_violate_idx,0], curve_exe[qddot_violate_idx,1], curve_exe[qddot_violate_idx,2], c=curve_exe[qddot_violate_idx,2], cmap='Greens',label='commanded points')
	plt.show()
