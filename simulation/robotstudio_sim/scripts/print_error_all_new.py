import numpy as np
from general_robotics_toolbox import *
from pandas import *
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from utils import *
from MotionSend import *

ms=MotionSend()

col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
# train_data = read_csv("../../../train_data/from_ge/Curve_in_base_frame2.csv", names=col_names)
# train_data = read_csv("../../../constraint_solver/single_arm/trajectory/curve_pose_opt/curve_pose_opt_cs.csv", names=col_names)
data = read_csv("../../../data/wood/Curve_in_base_frame.csv", names=col_names)

curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve_normal_x=data['direction_x'].tolist()
curve_normal_y=data['direction_y'].tolist()
curve_normal_z=data['direction_z'].tolist()
curve=np.vstack((curve_x, curve_y, curve_z)).T
curve_normal=np.vstack((curve_normal_x,curve_normal_y,curve_normal_z)).T

robot=abb6640(d=50)
d=50


# data_dir="fitting_output/threshold0.1/"
# speed={"v50":v50,"v500":v500,"v5000":v5000}
# zone={"fine":fine,"z1":z1,"z10":z10}
data_dir="greedy_output/wood_1/"
speed=['vmax']
zone=['z10']
max_error={}
max_error_idx={}
max_ori_error={}
jacobian_min_sing={}
jacobian_min_sing_idx={}

min_speed={}
max_speed={}
total_time={}
for s in speed:
	for z in zone: 
		act_speed=[]
		###read in curve_exe
		df = read_csv(data_dir+"curve_exe"+"_"+s+"_"+z+".csv")
		lam, curve_exe, curve_exe_R,curve_exe_js, act_speed, timestamp=ms.logged_data_analysis(robot,df)

		max_error[z,s],max_ori_error[z,s],max_error_idx[z,s]=calc_max_error_w_normal(curve_exe,curve,curve_exe_R[:,:,-1],curve_normal)


		jacobian_min_sing[z,s],jacobian_min_sing_idx[z,s]=find_j_min(robot,curve_exe_js)

		act_speed=np.array(act_speed)
		total_time[z,s]=timestamp[-1]-timestamp[0]
		min_speed[z,s]=np.min(act_speed[np.nonzero(act_speed)])
		max_speed[z,s]=np.max(act_speed)

table_names={"max_error":max_error,"max_error_idx":max_error_idx,"max_ori_error":max_ori_error,"total_time":total_time,"max_speed":max_speed,"min_speed":min_speed,\
			"jacobian_min_sing":jacobian_min_sing,"jacobian_min_sing_idx":jacobian_min_sing_idx}

data_all=[]
for table_name in table_names:
	data={}
	for s in speed:
		data[s]=[]
		for z in zone: 
			data[s].append(table_names[table_name][z,s])

	df = DataFrame(data, index=zone)
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