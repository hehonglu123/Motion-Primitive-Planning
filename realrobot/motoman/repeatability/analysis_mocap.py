import numpy as np
from general_robotics_toolbox import *
import sys
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from realrobot import *
from MotionSend_motoman import *
from lambda_calc import *

dataset='curve_1/'
solution_dir='baseline_motoman/'
data_dir='../../../data/'+dataset+solution_dir
cmd_dir=data_dir+'100L/'
curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
robot=robot_obj('MA2010_A0',def_path='../../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../../config/torch.csv',\
	pulse2deg_file_path='../../../config/MA2010_A0_pulse2deg_real.csv',d=50)

ms = MotionSend(robot)
breakpoints,primitives, p_bp,q_bp=ms.extract_data_from_cmd(cmd_dir+"command.csv")
p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp,extension_start=50,extension_end=50)

# speed=[100,150,200,250,300]
# num_runs=[4,5,6]
speed=[150]
num_runs=[5]
for N in num_runs:
	for v in speed:
		recorded_dir='recorded_data/'+dataset+'v%d_N%d/' % (v,N)

		ms=MotionSend(robot)
		###N run execute
		curve_exe_all=[]
		curve_exe_w_all=[]
		timestamp_all=[]
		total_time_all=[]
		for i in range(N):
			data=np.loadtxt(recorded_dir+'run_'+str(i)+'.csv',delimiter=',')
			
			##############################data analysis#####################################
			timestamp=data[:,0]
			curve_exe=data[:,1:4]
			curve_exe_w=data[:,4:7]
			lam=calc_lam_cs(curve_exe)
			curve_exe_R=w2R(curve_exe_w,np.eye(3))
			speed=get_speed(curve_exe,timestamp)

			###throw bad curves
			lam_temp, curve_exe_temp, curve_exe_R_temp, speed_temp, timestamp_temp=ms.chop_extension_mocap(curve_exe, curve_exe_R, speed, timestamp,curve[0,:3],curve[-1,:3],p_bp[0][0])
			curve_exe_w_temp=smooth_w(R2w(curve_exe_R_temp,np.eye(3)))

			error,angle_error=calc_all_error_w_normal(curve_exe_temp,curve[:,:3],curve_exe_R_temp[:,:,-1],curve[:,3:])
			
			######################################PLOT############################
			fig, ax1 = plt.subplots()
			ax2 = ax1.twinx()
			ax1.plot(lam_temp, speed_temp, 'g-', label='Speed')
			ax2.plot(lam_temp, error, 'b-',label='Error')
			ax2.plot(lam_temp, np.degrees(angle_error), 'y-',label='Normal Error')
			# ax2.axis(ymin=0,ymax=5)
			ax1.axis(ymin=0,ymax=1.2*v)

			ax1.set_xlabel('lambda (mm)')
			ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
			ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
			plt.title("Speed and Error Plot")
			h1, l1 = ax1.get_legend_handles_labels()
			h2, l2 = ax2.get_legend_handles_labels()
			ax1.legend(h1+h2, l1+l2, loc=1)

			plt.savefig(recorded_dir+'run_'+str(i))
			plt.clf()

			total_time_all.append(timestamp_temp[-1]-timestamp_temp[0])
			timestamp=timestamp-timestamp[0]
			curve_exe_all.append(curve_exe_temp)
			curve_exe_w_all.append(curve_exe_w_temp)
			timestamp_all.append(timestamp_temp)


		###trajectory outlier detection, based on chopped time
		curve_exe_all,curve_exe_w_all,timestamp_all=remove_traj_outlier_mocap(curve_exe_all,curve_exe_w_all,timestamp_all,total_time_all)
		

		###infer average curve from linear interplateion
		curve_exe_avg, curve_exe_w_avg, timestamp_d=average_curve_mocap(curve_exe_all,curve_exe_w_all,timestamp_all)
		curve_exe_R_avg=w2R(curve_exe_w_avg,np.eye(3))
		speed=get_speed(curve_exe_avg,timestamp_d)

		np.savetxt(recorded_dir+'/avg.csv',np.hstack((timestamp_d.reshape((-1,1)),curve_exe_avg,curve_exe_w_avg)),delimiter=',',comments='')


		#############################chop extension off##################################
		# lam, curve_exe, curve_exe_R, speed, timestamp=ms.chop_extension_mocap(curve_exe_avg, curve_exe_R_avg, speed, timestamp_d,curve[0,:3],curve[-1,:3],p_bp[0][0])
		# error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])

		lam=calc_lam_cs(curve_exe_avg)
		error,angle_error=calc_all_error_w_normal(curve_exe_avg,curve[:,:3],curve_exe_R_avg[:,:,-1],curve[:,3:])

		
		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		ax1.plot(lam, speed, 'g-', label='Speed')
		ax2.plot(lam, error, 'b-',label='Error')
		ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
		# ax2.axis(ymin=0,ymax=5)
		ax1.axis(ymin=0,ymax=1.2*v)

		ax1.set_xlabel('lambda (mm)')
		ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
		ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
		plt.title("Speed and Error Plot")
		h1, l1 = ax1.get_legend_handles_labels()
		h2, l2 = ax2.get_legend_handles_labels()
		ax1.legend(h1+h2, l1+l2, loc=1)

		plt.savefig(recorded_dir+'average')
		plt.clf()