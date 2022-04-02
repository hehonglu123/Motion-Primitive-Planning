from pandas import read_csv, DataFrame
import sys, copy
sys.path.append('../circular_Fit')
from toolbox_circular_fit import *
from abb_motion_program_exec_client import *
from robots_def import *
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from lambda_calc import *


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def blend_cs(q,curve,breakpoints,lam,primitives,robot):
	###blending in cartesian space
	#q:				joint configuration at each breakpoints
	#curve:			cartesian coordinates of full curve 50,000
	#breakpoints:	breakpoints
	#lam: 			path length
	#primitives:	primitive choices, L,C,J
	#robot:			robot kin tool def
	##return
	#q_blended: 	blended trajectory, N=10 points at each segments, 10 points within each blending

	N=10
	blending_radius=10
	act_breakpoints=copy.deepcopy(breakpoints)
	act_breakpoints[1:]=act_breakpoints[1:]-1
	sampled_path=[]
	sampled_path_ori=[]
	for i in range(len(primitives)):
		#get breakpiont pose
		start_pose=robot.fwd(q[i])
		end_pose=robot.fwd(q[i+1])
		#sample N points in each segment
		seg_length=lam[act_breakpoints[i+1]]-lam[act_breakpoints[i]]
		if i==0:
			lam_sampled_temp=np.linspace(lam[act_breakpoints[i]],lam[act_breakpoints[i+1]]-blending_radius,N)
		elif i==len(primitives)-1:
			lam_sampled_temp=np.linspace(lam[act_breakpoints[i]]+blending_radius,lam[act_breakpoints[i+1]],N)
		else:
			lam_sampled_temp=np.linspace(lam[act_breakpoints[i]]+blending_radius,lam[act_breakpoints[i+1]]-blending_radius,N)

		###primitive trajectory 
		if primitives[i]=='movel':
			a1,b1,c1=lineFromPoints([lam[act_breakpoints[i]],start_pose.p[0]],[lam[act_breakpoints[i+1]],end_pose.p[0]])
			a2,b2,c2=lineFromPoints([lam[act_breakpoints[i]],start_pose.p[1]],[lam[act_breakpoints[i+1]],end_pose.p[1]])
			a3,b3,c3=lineFromPoints([lam[act_breakpoints[i]],start_pose.p[2]],[lam[act_breakpoints[i+1]],end_pose.p[2]])
			seg_sample=np.vstack(((-a1*lam_sampled_temp-c1)/b1,(-a2*lam_sampled_temp-c2)/b2,(-a3*lam_sampled_temp-c3)/b3)).T

			
		elif primitives[i]=='movec':
			arc=arc_from_3point(start_pose.p,end_pose.p,curve((breakpoints[i+1]-breakpoints[i])/2),1000)
			idx=(lam_sampled_temp-lam[act_breakpoints[i]])/seg_length
			seg_sample=arc[idx.astype(int)*len(arc)]

		else:
			print('movej not implemented')
		sampled_path.extend(seg_sample)

		###breakpoints blending
		if i<len(primitives)-1:

	return q_blended
def blend_js(q,breakpoints,lam):
	#q: 			full 50,000 joints
	#breakpoints:	breakpoints
	#lam: 			path length
	##return
	#q_blended: 	blended trajectory with movej
	#merged_idx: 	merged blending if breakpoints too close
	#spl_list: list of spline coeff at each breakpoint

	blending=400###overestimated blending
	q_blended=copy.deepcopy(q)
	skip=False


	spl_list=[]
	merged_idx=[]
	for i in range(1,len(breakpoints)-1):
		spl_list_q=[]
		if skip:
			merged_idx[-1].append(breakpoints[i])
			skip=False
			continue

		merged_idx.append([breakpoints[i]])


		start_idx1	=breakpoints[i]-2*blending
		end_idx1	=breakpoints[i]-blending
		start_idx2	=breakpoints[i]+blending
		end_idx2	=breakpoints[i]+2*blending

		if i+1<len(breakpoints):
			if breakpoints[i+1]-breakpoints[i]<2*blending:
				skip=True
				start_idx1	=breakpoints[i]-blending
				end_idx1	=breakpoints[i]-1
				start_idx2	=breakpoints[i+1]+1
				end_idx2	=breakpoints[i+1]+blending

		for j in range(len(q[0])):
			spl = UnivariateSpline(np.hstack((lam[start_idx1:end_idx1],lam[start_idx2:end_idx2])),np.hstack((q[start_idx1:end_idx1,j],q[start_idx2:end_idx2,j])),k=5)
			q_blended[end_idx1:start_idx2,j]=spl(lam[end_idx1:start_idx2])
			spl_list_q.append(spl)
		spl_list.append(spl_list_q)

	spl_list=np.array(spl_list)
	return q_blended, spl_list, merged_idx



def blend_js2(q,breakpoints,lam):		
	###blend the trajectory assuming moveJ, given q at each breakpoint
	#q: 			joint configuration, at each breakpoint
	#breakpoints:	breakpoints
	#lam: 			path length
	##return
	#q_blended: 	blended trajectory with movej
	#dqdlam_list: 	list of qdot at each segment
	#spl_list: list of spline coeff at each breakpoint

	#######################################################create movej segs btw breakpoints###################################################
	total_points=50000
	lam2points=total_points/lam[-1]
	q_full=np.zeros((50000,len(q[0])))
	dqdlam_list=[]

	act_breakpoints=copy.deepcopy(breakpoints)
	act_breakpoints[1:]=act_breakpoints[1:]-1
	for i in range(1,len(breakpoints)):
		q_full[breakpoints[i-1]:breakpoints[i]]=np.linspace(q[i-1],q[i],num=breakpoints[i]-breakpoints[i-1],endpoint=False)
		dqdlam_list.append((q[i]-q[i-1])/(lam[act_breakpoints[i]]-lam[act_breakpoints[i-1]]))

	q_blended,spl_list, merged_idx=blend_js(q_full,breakpoints,lam)


	return q_blended,dqdlam_list, spl_list, merged_idx



def main():

	data_dir='../simulation/robotstudio_sim/scripts/fitting_output_new/threshold0.1/'

	data = read_csv(data_dir+'command.csv')
	breakpoints=np.array(data['breakpoints'].tolist())

	col_names=['J1', 'J2','J3', 'J4', 'J5', 'J6'] 
	data=read_csv(data_dir+'curve_fit_js.csv',names=col_names)
	q1=data['J1'].tolist()
	q2=data['J2'].tolist()
	q3=data['J3'].tolist()
	q4=data['J4'].tolist()
	q5=data['J5'].tolist()
	q6=data['J6'].tolist()
	curve_js=np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)

	col_names=['timestamp', 'cmd_num', 'J1', 'J2','J3', 'J4', 'J5', 'J6'] 
	data=read_csv(data_dir+'curve_exe_v1000_z10.csv',names=col_names)
	q1=data['J1'].tolist()[1:]
	q2=data['J2'].tolist()[1:]
	q3=data['J3'].tolist()[1:]
	q4=data['J4'].tolist()[1:]
	q5=data['J5'].tolist()[1:]
	q6=data['J6'].tolist()[1:]
	timestamp=np.array(data['timestamp'].tolist()[1:]).astype(float)
	cmd_num=np.array(data['cmd_num'].tolist()[1:]).astype(float)
	start_idx=np.where(cmd_num==5)[0][0]
	curve_js_act=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)[start_idx:])

	act_speed=[]
	lam_act=[0]
	curve_exe=[]
	for i in range(len(curve_exe_js)):
	    robot_pose=robot.fwd(curve_exe_js[i])
	    curve_exe.append(robot_pose.p)
	    if i>0:
	        lam_act.append(lam_act[-1]+np.linalg.norm(curve_exe[i]-curve_exe[i-1]))
	    try:
	        if timestamp[-1]!=timestamp[-2]:
	            act_speed.append(np.linalg.norm(curve_exe[-1]-curve_exe[-2])/timestep)
	            
	    except IndexError:
	        pass


	robot=abb6640(d=50)
	lam=calc_lam_js(curve_js,robot)

	lamdot_fit=calc_lamdot(curve_js,lam,robot,1)

	lamdot_act=calc_lamdot(curve_js_act,lam_act,robot,1)

	curve_blend_js, spl_list, merged_idx=blend_js(curve_js,breakpoints,lam)
	lamdot_blended=calc_lamdot(curve_blend_js,lam,robot,1)


	plt.plot(lam[1:],lamdot_fit, label='Fitting')
	plt.plot(lam[1:],lamdot_blended, label='Blended')
	plt.plot(lam_act[1:],act_speed, label='Actual Joints')
	plt.title("speed vs lambda")
	plt.ylabel('speed (mm/s)')
	plt.xlabel('lambda (mm)')
	plt.legend()
	plt.show()


	# df=DataFrame({'j1':curve_blend_js[:,0],'j2':curve_blend_js[:,1],'j3':curve_blend_js[:,2],'j4':curve_blend_js[:,3],'j5':curve_blend_js[:,4],'j6':curve_blend_js[:,5]})
	# df.to_csv(data_dir+'curve_blend_js.csv',header=False,index=False)


	# plt.plot(curve_js[:,0],label='original')
	# plt.plot(curve_blend_js[:,0],label='blended')
	# plt.title('Arbitrary Blending')
	# plt.xlabel('index')
	# plt.ylabel('q0 (rad)')
	# plt.legend()
	# plt.show()

def main2():

	data_dir='../simulation/robotstudio_sim/scripts/fitting_output_new/all_theta_opt_blended/'

	data = read_csv(data_dir+'command.csv')
	breakpoints=np.array(data['breakpoints'].tolist())

	col_names=['J1', 'J2','J3', 'J4', 'J5', 'J6'] 
	data=read_csv(data_dir+'curve_fit_js.csv',names=col_names)
	q1=data['J1'].tolist()
	q2=data['J2'].tolist()
	q3=data['J3'].tolist()
	q4=data['J4'].tolist()
	q5=data['J5'].tolist()
	q6=data['J6'].tolist()
	curve_js=np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)

	col_names=['timestamp', 'cmd_num', 'J1', 'J2','J3', 'J4', 'J5', 'J6'] 
	data=read_csv(data_dir+'curve_exe_vmax_z10.csv',names=col_names)
	q1=data['J1'].tolist()[1:]
	q2=data['J2'].tolist()[1:]
	q3=data['J3'].tolist()[1:]
	q4=data['J4'].tolist()[1:]
	q5=data['J5'].tolist()[1:]
	q6=data['J6'].tolist()[1:]
	timestamp=np.array(data['timestamp'].tolist()[1:]).astype(float)
	cmd_num=np.array(data['cmd_num'].tolist()[1:]).astype(float)
	start_idx=np.where(cmd_num==5)[0][0]
	curve_js_act=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)[start_idx:])
	timestep=np.average(timestamp[1:]-timestamp[:-1])

	robot=abb6640(d=50)

	act_speed=[]
	lam_act=[0]
	curve_exe=[]
	for i in range(len(curve_js_act)):
	    robot_pose=robot.fwd(curve_js_act[i])
	    curve_exe.append(robot_pose.p)
	    if i>0:
	        lam_act.append(lam_act[-1]+np.linalg.norm(curve_exe[i]-curve_exe[i-1]))
	    try:
	        if timestamp[-1]!=timestamp[-2]:
	            act_speed.append(np.linalg.norm(curve_exe[-1]-curve_exe[-2])/timestep)
	            
	    except IndexError:
	        pass



	lam=calc_lam_js(curve_js,robot)
	lam_act=calc_lam_js(curve_js_act,robot)

	lamdot_fit=calc_lamdot(curve_js,lam,robot,1)

	act_breakpoints=breakpoints
	act_breakpoints[1:]=act_breakpoints[1:]-1
	curve_blend_js,dqdlam_list,spl_list,merged_idx=blend_js2(curve_js[act_breakpoints],breakpoints,lam)


	lamdot_blended=calc_lamdot(curve_blend_js,lam,robot,1)
	lamdot_act=calc_lamdot(curve_js_act,lam_act,robot,1)
	dlam_max1,dlam_max2=est_lamdot(dqdlam_list,breakpoints,lam,spl_list,merged_idx,robot)
	lam_movej_seg_idx=(lam[breakpoints[:-1]]+lam[breakpoints[1:]])/2
	lam_est=np.hstack((lam_movej_seg_idx,lam[breakpoints[1:-1]]))

	lam_est, lamdot_est = zip(*sorted(zip(lam_est, np.hstack((dlam_max1,dlam_max2)))))

	plt.plot(lam[1:],lamdot_fit, label='Fitting')
	# plt.plot(lam[1:],lamdot_blended, label='Blended Estimated')
	plt.plot(lam_est,lamdot_est, label='Blended Estimated')
	plt.plot(lam_act[1:],act_speed, label='Actual Speed')
	plt.ylim(0,2100)
	plt.title("speed vs lambda")
	plt.ylabel('speed (mm/s)')
	plt.xlabel('lambda (mm)')
	plt.legend()
	plt.show()



if __name__ == '__main__':
	main2()
