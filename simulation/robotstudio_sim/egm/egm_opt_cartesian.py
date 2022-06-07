import numpy as np
import time, sys
from pandas import *
sys.path.append('egm_toolbox')
import rpi_abb_irc5

sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *
def main():
	robot=abb6640(d=50)

	egm = rpi_abb_irc5.EGM()

	dataset='wood/'
	data_dir='../../../data/'
	curve_js = read_csv(data_dir+dataset+'Curve_js.csv',header=None).values
	curve = read_csv(data_dir+dataset+'Curve_in_base_frame.csv',header=None).values
	curve_R=[]
	for q in curve_js:
		curve_R.append(robot.fwd(q).R)
	curve_R=np.array(curve_R)


	vd=150
	max_error_threshold=0.5
	
	lam=calc_lam_cs(curve[:,:3])
	ts=0.004

	steps=int((lam[-1]/vd)/ts)
	breakpoints=np.linspace(0.,len(curve_js)-1,num=steps).astype(int)
	curve_cmd_js=curve_js[breakpoints]
	curve_cmd=curve[breakpoints,:3]
	curve_cmd_R=curve_R[breakpoints]
	#################add extension#########################
	extension_num=150
	init_extension_js=np.linspace(curve_cmd_js[0]-extension_num*(curve_cmd_js[1]-curve_cmd_js[0]),curve_cmd_js[0],num=extension_num,endpoint=False)
	end_extension_js=np.linspace(curve_cmd_js[-1],curve_cmd_js[-1]+extension_num*(curve_cmd_js[-1]-curve_cmd_js[-2]),num=extension_num+1)[1:]

	init_extension_p=[]
	init_extension_R=[]
	for q in init_extension_js:
		pose_temp=robot.fwd(q)
		init_extension_p.append(pose_temp.p)
		init_extension_R.append(pose_temp.R)
	init_extension_p=np.array(init_extension_p)
	init_extension_R=np.array(init_extension_R)

	end_extension_p=[]
	end_extension_R=[]
	for q in end_extension_js:
		pose_temp=robot.fwd(q)
		end_extension_p.append(pose_temp.p)
		end_extension_R.append(pose_temp.R)
	end_extension_p=np.array(end_extension_p)
	init_extension_R=np.array(init_extension_R)

	curve_cmd_ext=np.vstack((init_extension_p,curve_cmd,end_extension_p))
	curve_cmd_R_ext=np.vstack((init_extension_R,curve_cmd_R,end_extension_R))

	max_error=999
	it=0
	while max_error>max_error_threshold:
		it+=1

		###move to start first
		print('moving to start point')

		res, state = egm.receive_from_robot(.1)
		q_cur=np.radians(state.joint_angles)
		pose_cur=robot.fwd(q_cur)
		num=int(np.linalg.norm(curve_cmd_ext[0]-pose_cur.p)/(1000*ts))
		curve2start=np.linspace(pose_cur.p,curve_cmd_ext[0],num=num)
		R2start=orientation_interp(pose_cur.R,curve_cmd_R_ext[0],num)
		quat2start=[]
		for R in R2start:
			quat2start.append(R2q(R))

		try:
			for i in range(len(curve2start)):
				while True:
					res_i, state_i = egm.receive_from_robot()
					if res_i:
						send_res = egm.send_to_robot_cart(curve2start[i], quat2start[i])
						break

			for i in range(500):
				while True:
					res_i, state_i = egm.receive_from_robot()
					if res_i:
						send_res = egm.send_to_robot_cart(curve_cmd_ext[0], R2q(curve_cmd_R_ext[0]))
						break

		except KeyboardInterrupt:
			raise

		curve_exe_js=[]
		timestamp=[]
		###traverse curve
		print('traversing trajectory')
		try:
			for i in range(len(curve_cmd_ext)):
				while True:
					res_i, state_i = egm.receive_from_robot()
					if res_i:
						send_res = egm.send_to_robot_cart(curve_cmd_ext[i], R2q(curve_cmd_R_ext[i]))
						#save joint angles
						curve_exe_js.append(np.radians(state_i.joint_angles))
						#TODO: replace with controller time
						timestamp.append(state_i.robot_message.header.tm)
						break
		except KeyboardInterrupt:
			raise
		
		timestamp=np.array(timestamp)/1000

		lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp,curve_exe_js)
		#############################chop extension off##################################
		start_idx=np.argmin(np.linalg.norm(curve[0,:3]-curve_exe,axis=1))
		end_idx=np.argmin(np.linalg.norm(curve[-1,:3]-curve_exe,axis=1))

		#make sure extension doesn't introduce error
		if np.linalg.norm(curve_exe[start_idx]-curve[0,:3])>0.5:
			start_idx+=1
		if np.linalg.norm(curve_exe[end_idx]-curve[-1,:3])>0.5:
			end_idx-=1

		curve_exe=curve_exe[start_idx:end_idx+1]
		curve_exe_R=curve_exe_R[start_idx:end_idx+1]
		speed=speed[start_idx:end_idx+1]
		# speed=replace_outliers(np.array(speed))
		lam=calc_lam_cs(curve_exe)

		##############################calcualte error########################################
		error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
		  
		

		curve_cmd_new=copy.deepcopy(curve_cmd)
		curve_cmd_R_new=copy.deepcopy(curve_cmd_R)
		curve_cmd_js=copy.deepcopy(curve_cmd_js)
		shift_vector=[]

		##############################Tweak breakpoints#################################################
		for i in range(len(curve_cmd)):
		    _,closest_exe_idx=calc_error(curve_cmd_new[i],curve_exe)        # find closest exe point
		    _,closest_curve_idx=calc_error(curve_exe[closest_exe_idx],curve[:,:3])  # find closest original curve point to closest exe point
		    d=(curve[closest_curve_idx,:3]-curve_exe[closest_exe_idx])/2           # shift vector
		    curve_cmd_new[i]+=d
		    ########orientation shift
		    R_temp=curve_exe_R[closest_exe_idx]@curve_R[closest_curve_idx].T
		    k,theta=R2rot(R_temp)
		    R_new=rot(k,-theta)@curve_R[closest_curve_idx]
		    curve_cmd_R_new[i]=R_new

		    shift_vector.append(d)
		
		shift_vector=np.array(shift_vector)
		###update ext cmd trajectory
		curve_cmd_ext[len(init_extension_p):-len(end_extension_p)]=curve_cmd_new
		curve_cmd_R_ext[len(init_extension_R):-len(end_extension_R)]=curve_cmd_R_new

		if it>5:
			##############################plot error#####################################
			fig, ax1 = plt.subplots()
			ax2 = ax1.twinx()
			ax1.plot(lam, speed, 'g-', label='Speed')
			ax2.plot(lam, error, 'b-',label='Error')
			ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')

			ax1.set_xlabel('lambda (mm)')
			ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
			ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
			plt.title("Speed and Error Plot")
			ax1.legend(loc=0)

			ax2.legend(loc=0)

			plt.legend()

			###########################plot for verification###################################
			plt.figure()
			ax = plt.axes(projection='3d')
			ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='gray',label='original')
			ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c='red',label='execution')
			ax.scatter3D(curve_cmd[:,0], curve_cmd[:,1], curve_cmd[:,2], c=curve_cmd[:,2], cmap='Greens',label='commanded points')

			ax.quiver(curve_cmd[:,0],curve_cmd[:,1],curve_cmd[:,2],shift_vector[:,0],shift_vector[:,1],shift_vector[:,2],length=1, normalize=True)

			plt.legend()
			plt.show()

		curve_cmd=curve_cmd_new

if __name__ == "__main__":
	main()