import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from scipy.signal import find_peaks

sys.path.append('../../toolbox')
from robots_def import *
from error_check import *
from MotionSend_motoman import *
from lambda_calc import *
from blending import *
from realrobot import *

sys.path.append('../')
from ilc_toolbox import *


def main():
	dataset='curve_2/'
	solution_dir='baseline_motoman/'
	data_dir="../../data/"+dataset+solution_dir
	cmd_dir="../../data/"+dataset+solution_dir+'greedy0.1L/'

	recorded_dir='curve2_baseline_100L_nPL/'

	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values


	multi_peak_threshold=0.4
	robot=robot_obj('MA2010_A0',def_path='../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun2.csv',\
    	pulse2deg_file_path='../../config/MA2010_A0_pulse2deg_real.csv',d=50)

	v=400
	z=None

	gamma_v_max=1
	gamma_v_min=0.2
	
	ms = MotionSend(robot)
	breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd(cmd_dir+'command.csv')
	###extension
	p_bp,q_bp=ms.extend(robot,q_bp,primitives,breakpoints,p_bp)
	# p_bp,q_bp,primitives,breakpoints=ms.extend2(robot,q_bp,primitives,breakpoints,p_bp)
	# breakpoints,primitives,p_bp,q_bp=ms.extract_data_from_cmd('curve2_pose_opt2_v1200/command.csv')

	
	###ilc toolbox def
	ilc=ilc_toolbox(robot,primitives)

	###TODO: extension fix start point, moveC support
	max_error_prev=999
	max_grad=False
	inserted_points=[]
	iteration=50

	N=5 	###N-run average
	curve_exe_js_all=[]
	timestamp_all=[]
	total_time_all=[]
	for i in range(N):
		data=np.loadtxt(recorded_dir+'run_'+str(i)+'.csv',delimiter=',')
		log_results=(data[:,0],data[:,2:8],data[:,1])
		##############################data analysis#####################################
		lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,log_results,realrobot=True)
		###throw bad curves
		lam_temp, curve_exe_temp, curve_exe_R_temp,curve_exe_js_temp, speed_temp, timestamp_temp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])
		
		total_time_all.append(timestamp_temp[-1]-timestamp_temp[0])
		timestamp=timestamp-timestamp[0]
		curve_exe_js_all.append(curve_exe_js)
		timestamp_all.append(timestamp)

	curve_exe_js_all,timestamp_all=remove_traj_outlier(curve_exe_js_all,timestamp_all,total_time_all)
	###infer average curve from linear interplateion
	curve_js_all_new, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)


	###calculat data with average curve
	lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp_d,avg_curve_js)
	#############################chop extension off##################################
	lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,avg_curve_js, speed, timestamp_d,curve[0,:3],curve[-1,:3])

	ms.write_data_to_cmd('recorded_data/command%i.csv'%i,breakpoints,primitives, p_bp,q_bp)

	##############################calcualte error########################################
	error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
	print(max(error),np.std(speed)/np.average(speed))

	#############################error peak detection###############################
	peaks,_=find_peaks(error,height=multi_peak_threshold,prominence=0.2,distance=20/(lam[int(len(lam)/2)]-lam[int(len(lam)/2)-1]))		###only push down peaks higher than height, distance between each peak is 20mm, threshold to filter noisy peaks
	
	if len(peaks)==0 or np.argmax(error) not in peaks:
		peaks=np.append(peaks,np.argmax(error))
	##############################plot error#####################################

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(lam, speed, 'g-', label='Speed')
	ax2.plot(lam, error, 'b-',label='Error')
	ax2.scatter(lam[peaks],error[peaks],label='peaks')
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

	plt.savefig('recorded_data/iteration_'+str(i))
	plt.clf()
	# plt.show()


	p_bp_old=copy.deepcopy(p_bp)
	q_bp_old=copy.deepcopy(q_bp)
	if max(error)<1.2*max_error_prev and not max_grad:
		print('ALL BPs ADJUSTMENT')
		##########################################adjust bp's toward error direction######################################
		error_bps_v,error_bps_w=ilc.get_error_direction(curve,p_bp,q_bp,curve_exe,curve_exe_R)
		p_bp, q_bp=ilc.update_error_direction(curve,p_bp,q_bp,error_bps_v,error_bps_w,gamma_v=0.5,gamma_w=0.1)
		for m in range(len(p_bp)):
			print(np.linalg.norm(q_bp[m][0]-q_bp_old[m][0]),np.linalg.norm(p_bp[m][0]-p_bp_old[m][0]))
	

if __name__ == "__main__":
	main()