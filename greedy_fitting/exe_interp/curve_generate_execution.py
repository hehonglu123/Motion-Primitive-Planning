import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import *
import sys, traceback
import numpy as np
sys.path.append('../data')
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



def curve_moveLC(q_prev,curve,start_R,end_R):
	curve_R_out=[]

	R_temp=np.dot(start_R.T,end_R)
	k,theta=R2rot(R_temp)
	###interpolate points between breakpoints
	for i in range(len(curve)):
		###interpolate orientation second
		angle=theta*i/float(len(curve))
		R=rot(k,angle)
		curve_R_out.append(np.dot(start_R,R))

	###convert to js
	curve_js_out=[]
	for i in range(len(curve)):
		try:
			q_all=np.array(inv(curve[i],curve_R_out[i]))
		except:
			traceback.print_exc()
			pass
		###choose inv_kin closest to previous joints
		try:
			temp_q=q_all-q_prev
			order=np.argsort(np.linalg.norm(temp_q,axis=1))
			curve_js_out.append(q_all[order[0]])
			q_prev=q_all[order[0]]

		except:
			traceback.print_exc()
			pass

	return np.array(curve_js_out)


def main():
	data = read_csv("../comparison/moveL+moveC/threshold1/curve_fit_backproj.csv")
	curve_x=data['x'].tolist()
	curve_y=data['y'].tolist()
	curve_z=data['z'].tolist()
	curve_fit=np.vstack((curve_x, curve_y, curve_z)).T
	breakpoints_out=[0,9064,15496,19997,25085,31478,38876,44918,47784,50007]

	col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
	data = read_csv("../data/from_ge/Curve_backproj_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T
	breakpoints=[1,9064,15495,19995,25082,31474,38871,44912,47777,49998]

	curve_js=[]
	q_init=np.array([0.627463700138299,0.17976842821744082,0.5196590573281621,1.6053098733278601,-0.8935105128511388,0.9174696574156079])
	start_R=direction2R(curve_direction[0],-curve[1]+curve[0])
	q_all=np.array(inv(curve_fit[0],start_R))
	###choose inv_kin closest to previous joints
	temp_q=q_all-q_init
	order=np.argsort(np.linalg.norm(temp_q,axis=1))
	q_init=q_all[order[0]]

	for i in range(len(breakpoints)-1):
		curve_js.append(curve_moveLC(q_init,curve_fit[breakpoints_out[i]:breakpoints_out[i+1]],direction2R(curve_direction[breakpoints[i]],-curve[breakpoints[i]+1]+curve[breakpoints[i]]),direction2R(curve_direction[breakpoints[i+1]],-curve[breakpoints[i+1]+1]+curve[breakpoints[i+1]])))

	curve_js=np.vstack(curve_js)
	df=DataFrame({'q0':curve_js[:,0],'q1':curve_js[:,1],'q2':curve_js[:,2],'q3':curve_js[:,3],'q4':curve_js[:,4],'q5':curve_js[:,5]})
	df.to_csv('Curve_js.csv',header=False,index=False)


if __name__ == "__main__":
	main()