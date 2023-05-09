import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from RobotRaconteur.Client import *

sys.path.append('../../toolbox')
from robot_def import *
from error_check import *
from MotionSend_motoman import *
from MocapPoseListener import *

def main():
	config_dir='../../config/'
	robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=50,  base_marker_config_file=config_dir+'MA2010_marker_config.yaml',\
		tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')
	

	mocap_url = 'rr+tcp://192.168.55.10:59823?service=optitrack_mocap'
	mocap_url = mocap_url
	mocap_cli = RRN.ConnectService(mocap_url)

	mpl_obj = MocapPoseListener(mocap_cli,[robot],collect_base_window=240)

	
	ms = MotionSend(robot)

	dataset='curve_1/'
	solution_dir='baseline_motoman/'
	# solution_dir='baseline_motoman/'

	data_dir='../../data/'+dataset+solution_dir
	cmd_dir=data_dir+'100L/'

	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
	lam_original=calc_lam_cs(curve[:,:3])

	

	speed=[300]

	for s in speed:
		breakpoints,primitives, p_bp,q_bp=ms.extract_data_from_cmd(cmd_dir+"command.csv")
		q_bp_end=q_bp[-1][0]
		# p_bp,q_bp,primitives,breakpoints = ms.extend2(robot, q_bp, primitives, breakpoints, p_bp)
		p_bp,q_bp = ms.extend(robot, q_bp, primitives, breakpoints, p_bp,extension_start=50,extension_end=50)
		zone=None
		mpl_obj.run_pose_listener()
	
		log_results = ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,s,zone)

		mpl_obj.stop_pose_listener()
		curve_exe,curve_exe_R,timestamp = mpl_obj.get_robots_traj()
		curve_exe = np.array(curve_exe[robot.robot_name])
		curve_exe_R = np.array(curve_exe_R[robot.robot_name])
		timestamp = np.array(timestamp[robot.robot_name])
		len_min=min(len(timestamp),len(curve_exe))
		curve_exe=curve_exe[:len_min]
		timestamp=timestamp[:len_min]

		###save results
		# curve_exe_w=R2w(curve_exe_R,np.eye(3))
		# np.savetxt('output.csv',np.hstack((timestamp.reshape((-1,1)),curve_exe, curve_exe_w)),delimiter=',',comments='')

		speed=get_speed(curve_exe,timestamp)
		lam, curve_exe, curve_exe_R, exe_speed, timestamp=ms.chop_extension_mocap(curve_exe, curve_exe_R, speed, timestamp,curve[0,:3],curve[-1,:3],p_bp[0][0])
	  
		##############################calcualte error########################################
		error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])

		print(np.std(exe_speed)/np.average(exe_speed))
		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		ax1.plot(lam, exe_speed, 'g-', label='Speed')
		ax2.plot(lam, error, 'b-',label='Error')
		ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
		# ax2.axis(ymin=0,ymax=4)
		ax1.axis(ymin=0,ymax=1.2*s)

		ax1.set_xlabel('lambda (mm)')
		ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
		ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
		plt.title("Speed and Error Plot")
		h1, l1 = ax1.get_legend_handles_labels()
		h2, l2 = ax2.get_legend_handles_labels()
		ax1.legend(h1+h2, l1+l2, loc=1)

		###plot breakpoints index
		breakpoints[1:]=breakpoints[1:]-1
		for bp in breakpoints:
			plt.axvline(x=lam_original[bp])

		plt.show()


if __name__ == "__main__":
	main()