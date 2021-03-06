from exe_comparison import *
import numpy as np
from general_robotics_toolbox import *
from pandas import *
import sys
from abb_motion_program_exec_client import *
sys.path.append('../../../toolbox')
from robots_def import *
from utils import *

data_dirs=['matlab_movej_0.1/','matlab_movej_0.5/','matlab_movej_1/']

ms = MotionSend()
speed={"v300":v300,"v500":v500,"vmax":vmax}
zone={"z10":z10}


col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
data = read_csv("../../../data/from_ge/Curve_in_base_frame.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve=np.vstack((curve_x, curve_y, curve_z)).T

robot=abb6640(d=50)



for data_dir_iter in data_dirs:
	data_dir="fitting_output_new/"+data_dir_iter
	for s in speed:
	    for z in zone: 
	        curve_exe_js=exe_from_file(ms,data_dir+"command.csv",data_dir+"curve_fit_js.csv",speed[s],zone[z])


	        f = open(data_dir+"curve_exe"+"_"+s+"_"+z+".csv", "w")
	        f.write(curve_exe_js)
	        f.close()
	
	print("execution done")



	max_error={}
	max_error_idx={}
	jacobian_min_sing={}
	jacobian_min_sing_idx={}

	min_speed={}
	max_speed={}
	total_time={}
	for s in speed:
		for z in zone: 
			act_speed=[]
			###read in curve_exe
			col_names=['timestamp', 'cmd_num', 'J1', 'J2','J3', 'J4', 'J5', 'J6'] 
			data = read_csv(data_dir+"curve_exe"+"_"+s+"_"+z+".csv",names=col_names)
			q1=data['J1'].tolist()[1:]
			q2=data['J2'].tolist()[1:]
			q3=data['J3'].tolist()[1:]
			q4=data['J4'].tolist()[1:]
			q5=data['J5'].tolist()[1:]
			q6=data['J6'].tolist()[1:]
			
			cmd_num=np.array(data['cmd_num'].tolist()[1:]).astype(float)
			start_idx=np.where(cmd_num==3)[0][0]
			timestamp=np.array(data['timestamp'].tolist()[1:]).astype(float)[start_idx:]
			curve_exe_js=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)[start_idx:])


			curve_exe=[]
			curve_exe_R=[]
			for i in range(len(curve_exe_js)):
				robot_pose=robot.fwd(curve_exe_js[i])
				curve_exe.append(robot_pose.p)
				curve_exe_R.append(robot_pose.R)
				try:
					if timestamp[-1]!=timestamp[-2]:
						act_speed.append(np.linalg.norm(curve_exe[-1]-curve_exe[-2])/(timestamp[-1]- timestamp[-2]))
				except IndexError:
					pass

			curve_exe_R=np.array(curve_exe_R)
			max_error[z,s],max_error_idx[z,s]=calc_max_error(curve_exe,curve)


			jacobian_min_sing[z,s],jacobian_min_sing_idx[z,s]=find_j_min(robot,curve_exe_js)

			act_speed=np.array(act_speed)
			total_time[z,s]=timestamp[-1]-timestamp[0]
			min_speed[z,s]=np.min(act_speed[np.nonzero(act_speed)])
			max_speed[z,s]=np.max(act_speed)

	table_names={"max_error":max_error,"max_error_idx":max_error_idx,"total_time":total_time,"max_speed":max_speed,"min_speed":min_speed,\
				"jacobian_min_sing":jacobian_min_sing,"jacobian_min_sing_idx":jacobian_min_sing_idx}

	data_all=[]
	for table_name in table_names:
		data={}
		for s in speed:
			data[s]=[]
			for z in zone: 
				data[s].append(table_names[table_name][z,s])

		df = DataFrame(data, index=list(zone.keys()))
		df.name=table_name
		print(table_name)
		print(df)
		data_all.append(df)


	writer = ExcelWriter(data_dir+'comparison.xlsx',engine='xlsxwriter')
	workbook=writer.book
	worksheet=workbook.add_worksheet('Result')
	writer.sheets['Result'] = worksheet


	for i in range(len(data_all)):
		try:
			worksheet.write_string(data_all[i-1].shape[0]+(data_all[i-1].shape[0]+2)*i, 0, data_all[i].name)
			data_all[i].to_excel(writer,sheet_name='Result',startrow=data_all[i-1].shape[0]+1+(data_all[i-1].shape[0]+2)*i , startcol=0)
		except IndexError:
			worksheet.write_string(0, 0, data_all[i].name)
			data_all[i].to_excel(writer,sheet_name='Result',startrow=1 , startcol=0)
	writer.save()
