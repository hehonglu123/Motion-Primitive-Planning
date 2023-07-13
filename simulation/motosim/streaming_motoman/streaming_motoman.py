import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from scipy.signal import find_peaks
from RobotRaconteur.Client import *

sys.path.append('../../../toolbox')
from robot_def import *
from error_check import *
from MotionSend_motoman import *
from lambda_calc import *
from blending import *
from realrobot import *
from StreamingSend import *
from MotionSend_motoman import *

def main():
	########################################################RR STREAMING########################################################
	RR_robot_sub = RRN.SubscribeService('rr+tcp://localhost:59945?service=robot')
	RR_robot_state = RR_robot_sub.SubscribeWire('robot_state')
	RR_robot = RR_robot_sub.GetDefaultClientWait(1)
	robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", RR_robot)
	halt_mode = robot_const["RobotCommandMode"]["halt"]
	position_mode = robot_const["RobotCommandMode"]["position_command"]
	RobotJointCommand = RRN.GetStructureType("com.robotraconteur.robotics.robot.RobotJointCommand",RR_robot)
	RR_robot.reset_errors()
	RR_robot.enable()
	RR_robot.command_mode = halt_mode
	time.sleep(0.1)
	RR_robot.command_mode = position_mode
	streaming_rate=125.
	point_distance=0.04		###STREAMING POINT INTERPOLATED DISTANCE
	

	dataset='curve_2/'
	solution_dir='baseline_motoman/'
	data_dir="../../../data/"+dataset+solution_dir

	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
	curve_js=read_csv(data_dir+"Curve_js.csv",header=None).values
	lam=calc_lam_cs(curve)

	robot=robot_obj('MA2010_A0',def_path='../../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../../config/torch.csv',\
		pulse2deg_file_path='../../../config/MA2010_A0_pulse2deg_real.csv',d=50)

	SS=StreamingSend(robot,RR_robot,RR_robot_state,RobotJointCommand,streaming_rate)
	ms = MotionSend(robot)

	lam_dense=np.linspace(0,lam[-1],num=int(lam[-1]/point_distance))
	curve_js_dense=interp1d(lam,curve_js,kind='cubic',axis=0)(lam_dense)

	v=300
	breakpoints=SS.get_breakpoints(lam_dense,v)
	curve_js_cmd=curve_js_dense[breakpoints]
	curve_js_cmd_ext=SS.add_extension_egm_js(curve_js_cmd,extension_start=50,extension_end=50)
	SS.jog2q(np.hstack((curve_js_cmd_ext[0],[np.pi/2,0,0,0,0,0,np.radians(-15),np.pi])))

	timestamp_recording,joint_recording=SS.traj_streaming(curve_js_cmd_ext,ctrl_joints=np.array([1,1,1,1,1,1,0,0,0,0,0,0,0,0]))
	
	###calculat data with average curve
	lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp_recording,joint_recording)
	#############################chop extension off##################################
	lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,joint_recording, speed, timestamp_recording,curve[0,:3],curve[-1,:3])
	error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
	
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(lam, speed, 'g-', label='Speed')
	ax2.plot(lam, error, 'b-',label='Error')
	ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
	ax2.axis(ymin=0,ymax=5)
	ax1.axis(ymin=0,ymax=1.2*v)

	ax1.set_xlabel('lambda (mm)')
	ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
	ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
	plt.title("Speed and Error Plot v=%f"%v)
	h1, l1 = ax1.get_legend_handles_labels()
	h2, l2 = ax2.get_legend_handles_labels()
	ax1.legend(h1+h2, l1+l2, loc=1)

	# plt.savefig('recorded_data/iteration_'+str(i))
	plt.show()
	
	
	np.savetxt('streaming_test/joint_recording_v%i.csv'%v,np.hstack((timestamp_recording.reshape(-1, 1),joint_recording)),delimiter=',')

if __name__ == '__main__':
	main()