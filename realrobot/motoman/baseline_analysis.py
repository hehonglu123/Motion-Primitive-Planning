import numpy as np
from general_robotics_toolbox import *
import sys
sys.path.append('../../toolbox')
from robots_def import *
from error_check import *
from realrobot import *
from MotionSend_motoman import *
from lambda_calc import *

dataset='curve_2/'
solution_dir='baseline_motoman/'
data_dir='../../data/'+dataset+solution_dir
cmd_dir=data_dir+'100L/'
curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
robot=robot_obj('MA2010_A0',def_path='../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun2.csv',\
	pulse2deg_file_path='../../config/MA2010_A0_pulse2deg.csv',d=50)

recorded_dir='recorded_data/iteration_0/'

ms=MotionSend(robot)
###N run execute
N=5
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
	error,angle_error=calc_all_error_w_normal(curve_exe_temp,curve[:,:3],curve_exe_R_temp[:,:,-1],curve[:,3:])
	######################################PLOT############################
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(lam_temp, speed_temp, 'g-', label='Speed')
	ax2.plot(lam_temp, error, 'b-',label='Error')
	ax2.plot(lam_temp, np.degrees(angle_error), 'y-',label='Normal Error')
	ax2.axis(ymin=0,ymax=5)
	# ax1.axis(ymin=0,ymax=1.2*v)

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
	curve_exe_js_all.append(curve_exe_js)
	timestamp_all.append(timestamp)






# print(total_time_all)
###trajectory outlier detection, based on chopped time
curve_exe_js_all,timestamp_all=remove_traj_outlier(curve_exe_js_all,timestamp_all,total_time_all)

###infer average curve from linear interplateion
curve_js_all_new, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)

lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp_d,avg_curve_js)
#############################chop extension off##################################
lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,avg_curve_js, speed, timestamp_d,curve[0,:3],curve[-1,:3])

error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(lam, speed, 'g-', label='Speed')
ax2.plot(lam, error, 'b-',label='Error')
ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
ax2.axis(ymin=0,ymax=5)
# ax1.axis(ymin=0,ymax=1.2*v)

ax1.set_xlabel('lambda (mm)')
ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
plt.title("Speed and Error Plot")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc=1)

plt.savefig(recorded_dir+'average')
plt.clf()



print(total_time_all)
###trajectory outlier detection, based on chopped time
curve_exe_js_all,timestamp_all=remove_traj_outlier(curve_exe_js_all,timestamp_all,total_time_all)