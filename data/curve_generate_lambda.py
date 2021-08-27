import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import *
import sys
import numpy as np
from cartesian2joint import direction2R
sys.path.append('../toolbox')
from robot_def import *
##################################generate equaly divided cartesian path for moveJ and moveL
def curve_moveJ(curve_js,d=0.0001):
	curve_js_out=[curve_js[0]]
	breakpoint_index=[0]
	curve_cartesian=[fwd(curve_js[0]).p]

	for i in range(len(curve_js)-1):
		move_direction=(curve_js[i+1]-curve_js[i])/np.linalg.norm(curve_js[i+1]-curve_js[i])
		while np.linalg.norm(curve_js_out[-1]-curve_js[i+1])>0.0001:
			curve_js_out.append(curve_js_out[-1]+d*move_direction)
			curve_cartesian.append(fwd(curve_js_out[-1]).p)

		breakpoint_index.append(len(curve_js_out)-1)
	print(breakpoint_index)

	return np.array(curve_js_out)



def curve_moveL(curve,curve_direction,d=0.1):
	curve_out=[curve[0]]
	curve_direction_out=[curve_direction[0]]
	breakpoint_index=[0]

	for i in range(len(curve)-1):
		move_direction=(curve[i+1]-curve_out[-1])/np.linalg.norm(curve[i+1]-curve_out[-1])
		rotate_axis=np.cross(curve_direction[i],curve_direction[i+1])
		rotate_axis=rotate_axis/np.linalg.norm(rotate_axis)
		rotate_angle = np.arccos(np.dot(curve_direction[i],curve_direction[i+1]))

		start_idx=len(curve_out)
		###interpolate position first
		while np.linalg.norm(curve_out[-1]-curve[i+1])>0.06:
			curve_out.append(curve_out[-1]+d*move_direction)
		###interpolate orientation second
		for j in range(start_idx,len(curve_out)):
			angle=rotate_angle*(j-start_idx)/(len(curve_out)-start_idx)
			R=rot(rotate_axis,angle)
			curve_direction_out.append(np.dot(R,curve_direction[i]))
		breakpoint_index.append(len(curve_out)-1)

	print(breakpoint_index)

	return np.array(curve_out),np.array(curve_direction_out)


def main():
	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("original/Curve_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("original/Curve_js.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T



	curve_out,curve_direction_out=curve_moveL(curve,curve_direction)

	# curve_js_out=curve_moveJ(curve_js)
	###output to csv
	# df=DataFrame({'q0':curve_js_out[:,0],'q1':curve_js_out[:,1],'q2':curve_js_out[:,2],'q3':curve_js_out[:,3],'q4':curve_js_out[:,4],'q5':curve_js_out[:,5]})
	# df.to_csv('execution/Curve_moveJ.csv',header=False,index=False)
	df=DataFrame({'x':curve_out[:,0],'y':curve_out[:,1], 'z':curve_out[:,2],'x_direction':curve_direction_out[:,0],'y_direction':curve_direction_out[:,1],'z_direction':curve_direction_out[:,2]})
	df.to_csv('execution/Curve_moveL.csv',header=False,index=False)

if __name__ == "__main__":
	main()