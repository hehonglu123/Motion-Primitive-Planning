import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys, traceback
from io import StringIO
from threading import Thread
sys.path.append('../')

from abb_motion_program_exec_client import *
from ilc_toolbox import *
from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *
from dual_arm import *

dataset='wood/'
data_dir="../../data/"+dataset
solution_dir=data_dir+'dual_arm/'+'diffevo3/'
cmd_dir=solution_dir+'50L/'
relative_path,robot1,robot2,base2_R,base2_p,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,data_dir,solution_dir,cmd_dir)


ms1 = MotionSend(robot1=robot1,robot2=robot2,base2_R=base2_R,base2_p=base2_p)
ms2=MotionSend(robot1=robot2, url="http://192.168.68.68:80")


df1=None
df2=None
def move_robot1():
	global df1, ms1, robot1,primitives1,breakpoints1,p_bp1,q_bp1,v1_all,z1_all

	try:
		logged_data= ms1.exec_motions(robot1,primitives1,breakpoints1,p_bp1,q_bp1,v1_all,z1_all)
		StringData=StringIO(logged_data)
		df1 = read_csv(StringData, sep =",")
	except:
		traceback.print_exc()

def move_robot2():
	global df2, ms2, robot2,primitives2,breakpoints2,p_bp2,q_bp2,v2_all,z2_all
	try:
		logged_data= ms2.exec_motions(robot2,primitives2,breakpoints2,p_bp2,q_bp2,v2_all,z2_all)
		StringData=StringIO(logged_data)
		df2 = read_csv(StringData, sep =",")
	except:
		traceback.print_exc()

def main():
	global ms1,ms2,df1, df2, robot1,primitives1,breakpoints1,p_bp1,q_bp1,v1_all,z1_all, robot2,primitives2,breakpoints2,p_bp2,q_bp2,v2_all,z2_all

	###extract data commands
	breakpoints1,primitives1,p_bp1,q_bp1=ms1.extract_data_from_cmd(cmd_dir+'command1.csv')
	breakpoints2,primitives2,p_bp2,q_bp2=ms2.extract_data_from_cmd(cmd_dir+'command2.csv')

	breakpoints1[1:]=breakpoints1[1:]-1
	breakpoints2[2:]=breakpoints2[2:]-1

	###ilc toolbox def
	ilc=ilc_toolbox([robot1,robot2],[primitives1,primitives2],base2_R,base2_p)

	z1_all=[z10]*len(breakpoints1)
	z2_all=[z10]*len(breakpoints2)

	###get lambda at each breakpoint
	lam_bp=lam_relative_path[np.append(breakpoints1[0],breakpoints1[1:]-1)]

	vd_relative=700

	s1_all,s2_all=calc_individual_speed(vd_relative,lam1,lam2,lam_relative_path,breakpoints1)
	v1_all=[]
	v2_all=[]
	for i in range(len(breakpoints1)):
		v1_all.append(speeddata(s1_all[i],9999999,9999999,999999))
		v2_all.append(speeddata(s2_all[i],9999999,9999999,999999))
	

	###extension
	p_bp1,q_bp1,p_bp2,q_bp2=ms1.extend_dual(robot1,p_bp1,q_bp1,primitives1,robot2,p_bp2,q_bp2,primitives2,breakpoints1,extension_start2=150,extension_end2=100)

	iteration=100
	for i in range(iteration):

		###move to start first 

		ms1.exec_motions(robot1,[primitives1[0]],[breakpoints1[0]],[p_bp1[0]],[q_bp1[0]],vmax,fine)
		ms2.exec_motions(robot2,[primitives2[0]],[breakpoints2[0]],[p_bp2[0]],[q_bp2[0]],vmax,fine)

		t1 = Thread(target=move_robot1)
		t2 = Thread(target=move_robot2)

		# start the threads
		t1.start()
		t2.start()

		# wait for the threads to complete
		t1.join()
		t2.join()

		df1.to_csv('recorded_data/r1_dual_iteration_'+str(i)+'.csv',header=True,index=False)
		df2.to_csv('recorded_data/r2_dual_iteration_'+str(i)+'.csv',header=True,index=False)

		lam_exe1, curve_exe1, curve_exe_R1,curve_exe_js1, speed1, timestamp1=ms1.logged_data_analysis(robot1,df1,realrobot=True)
		lam_exe2, curve_exe2, curve_exe_R2,curve_exe_js2, speed2, timestamp2=ms2.logged_data_analysis(robot2,df2,realrobot=True)

		if len(curve_exe_js1)>len(curve_exe_js2):
			print('size mismatch, padding now')
			# speed2=np.append(speed2,[0]*(len(curve_exe_js1)-len(curve_exe_js2)))
			curve_exe_R1=np.append(curve_exe_R1,[curve_exe_R1[-1]]*(len(curve_exe_js1)-len(curve_exe_js2)))
			curve_exe_js2=np.pad(curve_exe_js2,[(0,len(curve_exe_js1)-len(curve_exe_js2)),(0,0)], mode="edge")
			curve_exe2=np.pad(curve_exe2,[(0,len(curve_exe1)-len(curve_exe2)),(0,0)], mode="edge")
			
		elif len(curve_exe_js1)<len(curve_exe_js2):
			print('size mismatch, padding now')
			# speed1=np.append(speed1,[0]*(len(curve_exe_js2)-len(curve_exe_js1)))
			curve_exe_R2=np.append(curve_exe_R2,[curve_exe_R2[-1]]*(len(curve_exe_js2)-len(curve_exe_js1)))
			curve_exe_js1=np.pad(curve_exe_js1,[(0,len(curve_exe_js2)-len(curve_exe_js1)),(0,0)], mode="edge")
			curve_exe1=np.pad(curve_exe1,[(0,len(curve_exe2)-len(curve_exe1)),(0,0)], mode="edge")

		relative_path_exe,relative_path_exe_R=form_relative_path(robot1,robot2,curve_exe_js1,curve_exe_js2,base2_R,base2_p)
		lam=calc_lam_cs(relative_path_exe)
		speed=np.gradient(lam)/np.gradient(timestamp1)

		lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
				ms1.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp1, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])

		speed1=get_speed(curve_exe1,timestamp)
		speed2=get_speed(curve_exe2,timestamp)

		error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])


		fig, ax1 = plt.subplots()

		ax2 = ax1.twinx()
		ax1.plot(lam,speed, 'g-', label='Relative Speed')
		ax1.plot(lam,speed1, 'r-', label='TCP1 Speed')
		ax1.plot(lam,speed2, 'm-', label='TCP2 Speed')
		ax2.plot(lam, error, 'b-',label='Error')
		ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')

		ax1.set_xlabel('lambda (mm)')
		ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
		ax2.set_ylabel('Error (mm)', color='b')
		plt.title('Uncoordinated')

		h1, l1 = ax1.get_legend_handles_labels()
		h2, l2 = ax2.get_legend_handles_labels()
		ax1.legend(h1+h2, l1+l2, loc=1)
		plt.savefig('recorded_data/iteration_'+str(i))
		plt.clf()



		# ##########################################move towards error direction######################################
		# error_bps_v1,error_bps_w1,error_bps_v2,error_bps_w2=ilc.get_error_direction_dual(relative_path,p_bp1,q_bp1,p_bp2,q_bp2,relative_path_exe,relative_path_exe_R,curve_exe1,curve_exe_R1,curve_exe2,curve_exe_R2)

		# ###FIX ERROR_W!
		# error_bps_w1=np.zeros(error_bps_w1.shape)
		# error_bps_w2=np.zeros(error_bps_w2.shape)
		# p_bp1, q_bp1, p_bp2, q_bp2=ilc.update_error_direction_dual(relative_path,p_bp1,q_bp1,p_bp2,q_bp2,error_bps_v1,error_bps_w1,error_bps_v2,error_bps_w2,gamma_v=0.2,gamma_w=0.)

		
		###cmd speed adjustment
		speed_alpha=0.1

		for m in range(1,len(lam_bp)):
			###get segment average speed
			segment_avg=np.average(speed[np.argmin(np.abs(lam-lam_bp[m-1])):np.argmin(np.abs(lam-lam_bp[m]))])
			###cap above 100m/s for robot2
			s2_all[m]+=speed_alpha*(vd_relative-segment_avg)
			s2_all[m]=max(s2_all[m],100)
			v2_all[m]=speeddata(s2_all[m],9999999,9999999,999999)



if __name__ == "__main__":
	main()