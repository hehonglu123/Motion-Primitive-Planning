import numpy as np
import time, sys
from pandas import *

sys.path.append('../toolbox')
sys.path.append('../toolbox/egm_toolbox')

from robots_def import *
from error_check import *
from lambda_calc import *
from EGM_toolbox import *


def main():
	robot=abb6640(d=50)

	egm = rpi_abb_irc5.EGM()
	et=EGM_toolbox(egm,robot)
	idx_delay=int(et.delay/et.ts)

	dataset='wood/'
	data_dir='../data/'
	curve_js = read_csv(data_dir+dataset+'Curve_js.csv',header=None).values
	curve = read_csv(data_dir+dataset+'Curve_in_base_frame.csv',header=None).values
	curve_R=[]
	for q in curve_js:
		curve_R.append(robot.fwd(q).R)
	curve_R=np.array(curve_R)


	vd=250
	max_error_threshold=0.1
	
	lam=calc_lam_cs(curve[:,:3])

	steps=int((lam[-1]/vd)/et.ts)
	breakpoints=np.linspace(0.,len(curve_js)-1,num=steps).astype(int)
	curve_cmd_js=curve_js[breakpoints]
	curve_cmd=curve[breakpoints,:3]
	curve_cmd_R=curve_R[breakpoints]

	###extend whole tracking trajectory first
	extension_num1=100
	curve_cmd,curve_cmd_R=et.add_extension_egm_cartesian(curve_cmd,curve_cmd_R,extension_num=extension_num1)


	curve_cmd_w=R2w(curve_cmd_R)

	curve_d=copy.deepcopy(curve_cmd)
	curve_R_d=copy.deepcopy(curve_cmd_R)
	curve_w_d=copy.deepcopy(curve_cmd_w)
	
	extension_num=100

	max_error=999

	iteration=50
	adjust_weigt_it=2
	weight_adjusted=False
	for i in range(iteration):

		###add extension
		curve_cmd_ext,curve_cmd_R_ext=et.add_extension_egm_cartesian(curve_cmd,curve_cmd_R,extension_num=extension_num)

		###5 run execute
		curve_exe_js_all=[]
		timestamp_all=[]
		for r in range(5):
			###move to start first
			print('moving to start point')
			et.jog_joint_cartesian(curve_cmd_ext[0],curve_cmd_R_ext[0])
			
			###traverse the curve
			timestamp,curve_exe_js=et.traverse_curve_cartesian(curve_cmd_ext,curve_cmd_R_ext)

			timestamp=timestamp-timestamp[0]
			curve_exe_js_all.append(curve_exe_js)
			timestamp_all.append(timestamp)
			time.sleep(0.5)

		###infer average curve from linear interplateion
		curve_js_all_new, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)


		lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp_d[extension_num+idx_delay:-extension_num+idx_delay],avg_curve_js[extension_num+idx_delay:-extension_num+idx_delay])
		curve_exe_w=R2w(curve_exe_R,curve_R_d[0])
		error_distance,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])

		##############################ILC########################################
		error=curve_exe-curve_d
		error_distance=np.linalg.norm(error,axis=1)
		print('worst case error: ',np.max(error_distance))
		##add weights based on error
		weights_p=np.ones(len(error))


		error=error*weights_p[:, np.newaxis]
		error_flip=np.flipud(error)
		error_w=curve_exe_w-curve_w_d
		#add weights based on error_w
		# weights_w=np.linalg.norm(error_w,axis=1)
		# weights_w=(len(error)/4)*weights_w/weights_w.sum()
		weights_w=np.ones(len(error_w))

		error_w=error_w*weights_w[:, np.newaxis]
		error_w_flip=np.flipud(error_w)
		###calcualte agumented input
		curve_cmd_aug=curve_cmd+error_flip
		curve_cmd_w_aug=curve_cmd_w+error_w_flip
		curve_cmd_R_aug=w2R(curve_cmd_w_aug,curve_R_d[0])


		

		###add extension
		curve_cmd_ext_aug,curve_cmd_R_ext_aug=et.add_extension_egm_cartesian(curve_cmd_aug,curve_cmd_R_aug,extension_num=extension_num)
		###move to start first
		print('moving to start point')
		et.jog_joint_cartesian(curve_cmd_ext_aug[0],curve_cmd_R_ext_aug[0])
		###traverse the curve
		timestamp_aug,curve_exe_js_aug=et.traverse_curve_cartesian(curve_cmd_ext_aug,curve_cmd_R_ext_aug)

		_, curve_exe_aug, curve_exe_R_aug, _=logged_data_analysis(robot,timestamp_aug[extension_num+idx_delay:-extension_num+idx_delay],curve_exe_js_aug[extension_num+idx_delay:-extension_num+idx_delay])

		###get new error
		delta_new=curve_exe_aug-curve_exe
		delta_w_new=R2w(curve_exe_R_aug,curve_R_d[0])-curve_exe_w

		grad=np.flipud(delta_new)
		grad_w=np.flipud(delta_w_new)

		alpha1=1/np.sqrt(i+1)#0.5
		alpha2=1/np.sqrt(i+1)#1
		curve_cmd_new=curve_cmd-alpha1*grad
		curve_cmd_w-=alpha2*grad_w
		curve_cmd_R=w2R(curve_cmd_w,curve_R_d[0])

		curve_cmd=curve_cmd_new

		##############################plot error#####################################
		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		ax1.plot(lam[1:], speed, 'g-', label='Speed')
		ax2.plot(lam, error_distance, 'b-',label='Error')
		ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')

		ax1.set_xlabel('lambda (mm)')
		ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
		ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
		plt.title("Speed and Error Plot")
		ax1.legend(loc=0)

		ax2.legend(loc=0)

		plt.legend()

		###########################plot for verification###################################
		# plt.figure()
		# ax = plt.axes(projection='3d')
		# ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='gray',label='original')
		# ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c='red',label='execution')
		# ax.scatter3D(curve_cmd[:,0], curve_cmd[:,1], curve_cmd[:,2], c=curve_cmd[:,2], cmap='Greens',label='commanded points')
		# ax.scatter3D(curve_cmd_new[:,0], curve_cmd_new[:,1], curve_cmd_new[:,2], c=curve_cmd_new[:,2], cmap='Blues',label='new commanded points')
		# plt.legend()


		# plt.show()
		plt.savefig('iteration_ '+str(i))
		plt.clf()
if __name__ == "__main__":
	main()