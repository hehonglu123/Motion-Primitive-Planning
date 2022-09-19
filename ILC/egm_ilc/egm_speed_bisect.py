import numpy as np
import time, sys
from pandas import *

from robots_def import *
from error_check import *
from lambda_calc import *
from EGM_toolbox import *
from realrobot import *

def main():
	robot=abb6640(d=50)

	egm = rpi_abb_irc5.EGM()
	et=EGM_toolbox(egm,robot)
	idx_delay=int(et.delay/et.ts)

	dataset='from_NX/'
	solution_dir='baseline/'
	data_dir="../../data/"+dataset+solution_dir
	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
	curve_js = read_csv(data_dir+"Curve_js.csv",header=None).values

	curve_R=[]
	for q in curve_js:
		curve_R.append(robot.fwd(q).R)
	curve_R=np.array(curve_R)


	
	error_threshold=0.5
	angle_threshold=np.radians(3)
	
	lam=calc_lam_cs(curve[:,:3])

	extension_num=100

	v=150
	v_prev=200
	v_prev_possible=100
	i=0
	while True:
		steps=int((lam[-1]/v)/et.ts)
		breakpoints=np.linspace(0.,len(curve_js)-1,num=steps).astype(int)
		curve_cmd_js=curve_js[breakpoints]
		curve_cmd=curve[breakpoints,:3]
		curve_cmd_R=curve_R[breakpoints]
		curve_cmd_w=R2w(curve_cmd_R)
	
	


		###add extension
		curve_cmd_ext,curve_cmd_R_ext=et.add_extension_egm_cartesian(curve_cmd,curve_cmd_R,extension_num=extension_num)

		###5 run execute
		curve_js_all_new,avg_curve_js, timestamp_d=average_5_egm_car_exe(et,curve_cmd_ext,curve_cmd_R_ext)

		df=DataFrame({'timestamp':timestamp_d,'q0':avg_curve_js[:,0],'q1':avg_curve_js[:,1],'q2':avg_curve_js[:,2],'q3':avg_curve_js[:,3],'q4':avg_curve_js[:,4],'q5':avg_curve_js[:,5]})
		df.to_csv('recorded_data/iteration'+str(i)+'.csv',header=False,index=False)

		lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp_d[extension_num+idx_delay:-extension_num+idx_delay],avg_curve_js[extension_num+idx_delay:-extension_num+idx_delay])

		error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
		print('v',v)
		print('std speed: ',np.std(speed),'angle error: ',max(angle_error) ,'worst case error: ',np.max(error))


		v_prev_temp=v
		if np.max(error)>error_threshold or np.std(speed)>np.average(speed)/20 or max(angle_error)>angle_threshold:
			v-=abs(v_prev-v)/2
		else:
			v_prev_possible=v
			#stop condition
			if error_threshold-np.max(error)<0.02:
				break   
			v+=abs(v_prev-v)/2

		v_prev=v_prev_temp
		i+=1

if __name__ == '__main__':
	main()