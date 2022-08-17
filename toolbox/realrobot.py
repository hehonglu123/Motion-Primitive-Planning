

def average_5_exe(ms,robot,primitives,breakpoints,p_bp,q_bp,s,z,log_path=''):
	###5 run execute
	curve_exe_all=[]
	curve_exe_js_all=[]
	timestamp_all=[]
	total_time_all=[]

	for r in range(5):
		logged_data=ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,s,z)
		###save 5 runs
		if len(log_path)>0:
			# Write log csv to file
			with open(log_path+'/run_'+str(r)+'.csv',"w") as f:
			    f.write(logged_data)

		StringData=StringIO(logged_data)
		df = read_csv(StringData, sep =",")
		##############################data analysis#####################################
		lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)

		###throw bad curves
		_, _, _,_, _, timestamp_temp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve[0,:3],curve[-1,:3])
		total_time_all.append(timestamp_temp[-1]-timestamp_temp[0])

		timestamp=timestamp-timestamp[0]

		curve_exe_all.append(curve_exe)
		curve_exe_js_all.append(curve_exe_js)
		timestamp_all.append(timestamp)

	###trajectory outlier detection, based on chopped time
	curve_exe_all,curve_exe_js_all,timestamp_all=remove_traj_outlier(curve_exe_all,curve_exe_js_all,timestamp_all,total_time_all)

	###infer average curve from linear interplateion
	curve_js_all_new, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)

	return curve_js_all_new, avg_curve_js, timestamp_d