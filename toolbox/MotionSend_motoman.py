import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from dx200_motion_program_exec_client import *
from robots_def import *
from error_check import *
from toolbox_circular_fit import *
from lambda_calc import *


name_map={'MA2010_A0':('RB1',0,6),'MA1440_A0':('RB2',6,12),'D500B':('ST1',12,14)}

class MotionSend(object):
	def __init__(self,robot1,robot2=None,IP='192.168.1.31') -> None:
		###SPECIFY TOOL NUMBER HERE
		if robot2:
			self.client=MotionProgramExecClient(ROBOT_CHOICE=name_map[robot1.robot_name][0],ROBOT_CHOICE2=name_map[robot2.robot_name][0],pulse2deg=robot1.pulse2deg,pulse2deg_2=robot2.pulse2deg)
		else:
			self.client=MotionProgramExecClient(ROBOT_CHOICE=name_map[robot1.robot_name][0],pulse2deg=robot1.pulse2deg, tool_num = 11)



	def jog_joint(self,q):
		mp = MotionProgram()
		mp.MoveAbsJ(self.moveJ_target(q),v500,fine)
	

	def exec_motions(self,robot,primitives,breakpoints,p_bp,q_bp,speed,zone=None):
		for i in range(len(primitives)):
			if 'movel' in primitives[i]:
				if type(speed) is list:
					if type(zone) is list:
						self.client.MoveL(np.degrees(q_bp[i][0]), speed[i], zone[i])
					else:
						self.client.MoveL(np.degrees(q_bp[i][0]), speed[i], zone)
				else:
					if type(zone) is list:
						self.client.MoveL(np.degrees(q_bp[i][0]), speed, zone[i])
					else:
						self.client.MoveL(np.degrees(q_bp[i][0]), speed, zone)
				

			elif 'movec' in primitives[i]:

				if type(speed) is list:
					if type(zone) is list:
						mp.MoveC(np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed[i],zone[i])
					else:
						mp.MoveC(np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed[i],zone)
				else:
					if type(zone) is list:
						mp.MoveC(np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed,zone[i])
					else:
						mp.MoveC(np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed,zone)

			else:
				###special case for motoman, speed is in %
				if i==0 and type(speed) is not list:

					self.client.MoveJ(np.degrees(q_bp[i][0]), 5)
					self.client.setWaitTime(1)
					self.client.MoveJ(np.degrees(q_bp[i][0]), 1)
					self.client.setWaitTime(0.1)

				else:

					if type(speed) is list:
						if type(zone) is list:
							self.client.MoveJ(np.degrees(q_bp[i][0]), speed[i], zone[i])
						else:
							self.client.MoveJ(np.degrees(q_bp[i][0]), speed[i], zone)
					else:
						if type(zone) is list:
							self.client.MoveJ(np.degrees(q_bp[i][0]), speed, zone[i])
						else:
							self.client.MoveJ(np.degrees(q_bp[i][0]), speed, zone)

		timestamp, joint_recording, job_line,job_step = self.client.execute_motion_program()
		return (timestamp, joint_recording[:,name_map[robot.robot_name][1]:name_map[robot.robot_name][2]], job_line,job_step)

	def exe_from_file(self,robot,filename,speed,zone=None):
		breakpoints,primitives, p_bp,q_bp=self.extract_data_from_cmd(filename)
		return self.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,speed,zone)


	def extend(self,robot,q_bp,primitives,breakpoints,p_bp,extension_start=100,extension_end=100):
		p_bp_extended=copy.deepcopy(p_bp)
		q_bp_extended=copy.deepcopy(q_bp)
		###initial point extension
		pose_start=robot.fwd(q_bp[0][0])
		p_start=pose_start.p
		R_start=pose_start.R
		pose_end=robot.fwd(q_bp[1][-1])
		p_end=pose_end.p
		R_end=pose_end.R
		if 'movel' in primitives[1]:
			#find new start point
			slope_p=p_end-p_start
			slope_p=slope_p/np.linalg.norm(slope_p)
			p_start_new=p_start-extension_start*slope_p        ###extend 5cm backward

			#find new start orientation
			k,theta=R2rot(R_end@R_start.T)
			theta_new=-extension_start*theta/np.linalg.norm(p_end-p_start)
			R_start_new=rot(k,theta_new)@R_start

			#solve invkin for initial point
			p_bp_extended[0][0]=p_start_new
			q_bp_extended[0][0]=car2js(robot,q_bp[0][0],p_start_new,R_start_new)[0]

		elif 'movec' in primitives[1]:
			#define circle first
			pose_mid=robot.fwd(q_bp[1][0])
			p_mid=pose_mid.p
			R_mid=pose_mid.R

			center, radius=circle_from_3point(p_start,p_end,p_mid)

			#find desired rotation angle
			angle=extension_start/radius

			#find new start point
			plane_N=np.cross(p_end-center,p_start-center)
			plane_N=plane_N/np.linalg.norm(plane_N)
			R_temp=rot(plane_N,angle)
			p_start_new=center+R_temp@(p_start-center)

			#modify mid point to be in the middle of new start and old end (to avoid RS circle uncertain error)
			modified_bp=arc_from_3point(p_start_new,p_end,p_mid,N=3)
			p_bp_extended[1][0]=modified_bp[1]

			#find new start orientation
			k,theta=R2rot(R_end@R_start.T)
			theta_new=-extension_start*theta/np.linalg.norm(p_end-p_start)
			R_start_new=rot(k,theta_new)@R_start

			#solve invkin for initial point
			p_bp_extended[0][0]=p_start_new
			q_bp_extended[0][0]=car2js(robot,q_bp[0][0],p_start_new,R_start_new)[0]


		else:
			#find new start point
			J_start=robot.jacobian(q_bp[0][0])
			qdot=q_bp[0][0]-q_bp[1][0]
			v=np.linalg.norm(J_start[3:,:]@qdot)
			t=extension_start/v
			q_bp_extended[0][0]=q_bp[0][0]+qdot*t
			p_bp_extended[0][0]=robot.fwd(q_bp_extended[0][0]).p

		###end point extension
		pose_start=robot.fwd(q_bp[-2][-1])
		p_start=pose_start.p
		R_start=pose_start.R
		pose_end=robot.fwd(q_bp[-1][-1])
		p_end=pose_end.p
		R_end=pose_end.R

		if 'movel' in primitives[-1]:
			#find new end point
			slope_p=(p_end-p_start)/np.linalg.norm(p_end-p_start)
			p_end_new=p_end+extension_end*slope_p        ###extend 5cm backward

			#find new end orientation
			k,theta=R2rot(R_end@R_start.T)
			slope_theta=theta/np.linalg.norm(p_end-p_start)
			R_end_new=rot(k,extension_end*slope_theta)@R_end

			#solve invkin for end point
			q_bp_extended[-1][0]=car2js(robot,q_bp[-1][0],p_end_new,R_end_new)[0]
			p_bp_extended[-1][0]=p_end_new


		elif  'movec' in primitives[-1]:
			#define circle first
			pose_mid=robot.fwd(q_bp[-1][0])
			p_mid=pose_mid.p
			R_mid=pose_mid.R
			center, radius=circle_from_3point(p_start,p_end,p_mid)

			#find desired rotation angle
			angle=extension_end/radius

			#find new end point
			plane_N=np.cross(p_start-center,p_end-center)
			plane_N=plane_N/np.linalg.norm(plane_N)
			R_temp=rot(plane_N,angle)
			p_end_new=center+R_temp@(p_end-center)

			#modify mid point to be in the middle of new end and old start (to avoid RS circle uncertain error)
			modified_bp=arc_from_3point(p_start,p_end_new,p_mid,N=3)
			p_bp_extended[-1][0]=modified_bp[1]

			#find new end orientation
			k,theta=R2rot(R_end@R_start.T)
			theta_new=extension_end*theta/np.linalg.norm(p_end-p_start)
			R_end_new=rot(k,theta_new)@R_end

			#solve invkin for end point
			q_bp_extended[-1][-1]=car2js(robot,q_bp[-1][-1],p_end_new,R_end_new)[0]
			p_bp_extended[-1][-1]=p_end_new   #midpoint not changed

		else:
			#find new end point
			J_end=robot.jacobian(q_bp[-1][0])
			qdot=q_bp[-1][0]-q_bp[-2][0]
			v=np.linalg.norm(J_end[3:,:]@qdot)
			t=extension_end/v
			
			q_bp_extended[-1][0]=q_bp[-1][-1]+qdot*t
			p_bp_extended[-1][0]=robot.fwd(q_bp_extended[-1][-1]).p

		return p_bp_extended,q_bp_extended

	def extend2(self,robot,q_bp,primitives,breakpoints,p_bp,extension_start=100,extension_end=100):
		##########################extend by adding another segment, adjust bp/primitives by reference
		###initial point extension
		pose_start=robot.fwd(q_bp[0][0])
		p_start=pose_start.p
		R_start=pose_start.R
		pose_end=robot.fwd(q_bp[1][-1])
		p_end=pose_end.p
		R_end=pose_end.R

		#find new start point
		slope_p=p_end-p_start
		slope_p=slope_p/np.linalg.norm(slope_p)
		p_start_new=p_start-extension_start*slope_p        ###extend 5cm backward

		#find new start orientation
		k,theta=R2rot(R_end@R_start.T)
		theta_new=-extension_start*theta/np.linalg.norm(p_end-p_start)
		R_start_new=rot(k,theta_new)@R_start

		#solve invkin for initial point
		p_bp.insert(0,[p_start_new])
		q_bp.insert(0,[car2js(robot,q_bp[0][0],p_start_new,R_start_new)[0]])
		primitives.insert(1,'movel_fit')
		breakpoints=np.insert(breakpoints,0,0)



		###end point extension
		pose_start=robot.fwd(q_bp[-2][-1])
		p_start=pose_start.p
		R_start=pose_start.R
		pose_end=robot.fwd(q_bp[-1][-1])
		p_end=pose_end.p
		R_end=pose_end.R

		#find new end point
		slope_p=(p_end-p_start)/np.linalg.norm(p_end-p_start)
		p_end_new=p_end+extension_end*slope_p        ###extend 5cm backward

		#find new end orientation
		k,theta=R2rot(R_end@R_start.T)
		slope_theta=theta/np.linalg.norm(p_end-p_start)
		R_end_new=rot(k,extension_end*slope_theta)@R_end

		#solve invkin for end point
		p_bp.append([p_end_new])
		q_bp.append([car2js(robot,q_bp[-1][0],p_end_new,R_end_new)[0]])
		primitives.append('movel_fit')
		breakpoints=np.append(breakpoints,breakpoints[-1])
		return p_bp,q_bp,primitives,breakpoints


	def extract_data_from_cmd(self,filename):
		data = read_csv(filename)
		breakpoints=np.array(data['breakpoints'].tolist())
		primitives=data['primitives'].tolist()
		points=data['p_bp'].tolist()
		qs=data['q_bp'].tolist()

		p_bp=[]
		q_bp=[]
		for i in range(len(breakpoints)):
			if 'movel' in primitives[i]:
				point=extract_points(primitives[i],points[i])
				p_bp.append([point])
				q=extract_points(primitives[i],qs[i])
				q_bp.append([q])


			elif 'movec' in primitives[i]:
				point1,point2=extract_points(primitives[i],points[i])
				p_bp.append([point1,point2])
				q1,q2=extract_points(primitives[i],qs[i])
				q_bp.append([q1,q2])

			else:
				point=extract_points(primitives[i],points[i])
				p_bp.append([point])
				q=extract_points(primitives[i],qs[i])
				q_bp.append([q])

		return breakpoints,primitives, p_bp,q_bp

	def write_data_to_cmd(self,filename,breakpoints,primitives, p_bp,q_bp):
		p_bp_new=[]
		q_bp_new=[]
		for i in range(len(breakpoints)):
			if len(p_bp[i])==2:
				p_bp_new.append([np.array(p_bp[i][0]),np.array(p_bp[i][1])])
				q_bp_new.append([np.array(q_bp[i][0]),np.array(q_bp[i][1])])
			else:
				p_bp_new.append([np.array(p_bp[i][0])])
				q_bp_new.append([np.array(q_bp[i][0])])
		df=DataFrame({'breakpoints':breakpoints,'primitives':primitives,'p_bp':p_bp_new,'q_bp':q_bp_new})
		df.to_csv(filename,header=True,index=False)

	def parse_logged_data(self,log_results):		###convert packet to timestamp and joint angle
		return log_results[0], log_results[1], log_results[2].astype(int)

	def logged_data_analysis(self,robot,log_results,realrobot=True):
		timestamp, curve_exe_js,job_line=self.parse_logged_data(log_results)

		#find closest to 5 cmd_num
		idx = np.absolute(job_line-5).argmin()
		start_idx=np.where(job_line==job_line[idx])[0][0]

		timestamp=timestamp[start_idx:]
		curve_exe_js=curve_exe_js[start_idx:]

		###filter noise
		timestamp, curve_exe_js=lfilter(timestamp, curve_exe_js)

		speed=[0]
		lam=[0]
		curve_exe=[]
		curve_exe_R=[]
		for i in range(len(curve_exe_js)):
			robot_pose=robot.fwd(curve_exe_js[i],qlim_override=True)
			curve_exe.append(robot_pose.p)
			curve_exe_R.append(robot_pose.R)
			if i>0:
				lam.append(lam[-1]+np.linalg.norm(curve_exe[i]-curve_exe[i-1]))
			try:
				if timestamp[i-1]!=timestamp[i] and np.linalg.norm(curve_exe_js[i-1]-curve_exe_js[i])!=0:
					speed.append(np.linalg.norm(curve_exe[-1]-curve_exe[-2])/(timestamp[i]-timestamp[i-1]))
				else:
					speed.append(speed[-1])      
			except IndexError:
				pass

		speed=moving_average(speed,padding=True)
		return np.array(lam), np.array(curve_exe), np.array(curve_exe_R),np.array(curve_exe_js), np.array(speed), timestamp-timestamp[0]

	

	def chop_extension(self,curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,p_start,p_end):
		start_idx=np.argmin(np.linalg.norm(p_start-curve_exe,axis=1))
		end_idx=np.argmin(np.linalg.norm(p_end-curve_exe,axis=1))

		#make sure extension doesn't introduce error
		if np.linalg.norm(curve_exe[start_idx]-p_start)>0.3:
			start_idx+=1
		if np.linalg.norm(curve_exe[end_idx]-p_end)>0.3:
			end_idx-=1

		curve_exe=curve_exe[start_idx:end_idx+1]
		curve_exe_js=curve_exe_js[start_idx:end_idx+1]
		curve_exe_R=curve_exe_R[start_idx:end_idx+1]
		speed=speed[start_idx:end_idx+1]
		lam=calc_lam_cs(curve_exe)

		return lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp[start_idx:end_idx+1]-timestamp[start_idx]

	
	def calc_robot2_q_from_blade_pose(self,blade_pose,base2_R,base2_p):
		R2=base2_R.T@blade_pose[:3,:3]
		p2=-base2_R.T@(base2_p-blade_pose[:3,-1])

		return self.robot2.inv(p2,R2)[0]


def main():
	return
if __name__ == "__main__":
	main()