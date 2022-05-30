import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from abb_motion_program_exec_client import *
from robots_def import *
from error_check import *
sys.path.append('../toolbox')
from toolbox_circular_fit import *

quatR = R2q(rot([0,1,0],math.radians(30)))
class MotionSend(object):
    def __init__(self,robot1=abb1200(),robot2=abb6640(d=50),tool1=tooldata(True,pose([50,0,450],[quatR[0],quatR[1],quatR[2],quatR[3]]),loaddata(1,[0,0,0.001],[1,0,0,0],0,0,0)),tool2=tooldata(True,pose([75,0,493.30127019],[quatR[0],quatR[1],quatR[2],quatR[3]]),loaddata(1,[0,0,0.001],[1,0,0,0],0,0,0))) -> None:
        ###robot1: 1200
        ###robot2: 6640 with d=50 fake link
        
        self.client = MotionProgramExecClient()

        ###with fake link
        self.robot1=robot1
        self.robot2=robot2
        
        self.tool1 = tool1
        self.tool2 = tool2
        

    def moveL_target(self,robot,q,point):
        quat=R2q(robot.fwd(q).R)
        cf=quadrant(q)
        robt = robtarget([point[0], point[1], point[2]], [ quat[0], quat[1], quat[2], quat[3]], confdata(cf[0],cf[1],cf[2],cf[3]),[9E+09]*6)
        return robt
    
    def moveC_target(self,robot,q1,q2,point1,point2):
        quat1=R2q(robot.fwd(q1).R)
        cf1=quadrant(q1)
        quat2=R2q(robot.fwd(q2).R)
        cf2=quadrant(q2)
        robt1 = robtarget([point1[0], point1[1], point1[2]], [ quat1[0], quat1[1], quat1[2], quat1[3]], confdata(cf1[0],cf1[1],cf1[2],cf1[3]),[0]*6)
        robt2 = robtarget([point2[0], point2[1], point2[2]], [ quat2[0], quat2[1], quat2[2], quat2[3]], confdata(cf2[0],cf2[1],cf2[2],cf2[3]),[0]*6)
        return robt1, robt2

    def moveJ_target(self,q):
        q = np.rad2deg(q)
        jointt = jointtarget([q[0],q[1],q[2],q[3],q[4],q[5]],[0]*6)
        return jointt

    def exec_motions(self,robot,primitives,breakpoints,points,curve_js,speed,zone):
        mp = MotionProgram(tool=self.tool2)
        
        for i in range(len(primitives)):
            ###force last point to be fine
            # if i==len(primitives)-1:
            #     zone=fine
            motion = primitives[i]
            if motion == 'movel_fit':
                robt = self.moveL_target(robot,curve_js[breakpoints[i]],points[i])
                mp.MoveL(robt,speed,zone)

            elif motion == 'movec_fit':
                robt1, robt2 = self.moveC_target(robot,curve_js[breakpoints[i-1]],curve_js[breakpoints[i]],points[i][0],points[i][1])
                mp.MoveC(robt1,robt2,speed,zone)

            else: # movej_fit
                jointt = self.moveJ_target(points[i])
                if i==0:
                    mp.MoveAbsJ(jointt,v500,fine)
                    mp.WaitTime(1)
                    mp.MoveAbsJ(jointt,v500,fine)
                    mp.WaitTime(0.1)
                else:
                    mp.MoveAbsJ(jointt,speed,zone)
        ###add sleep at the end to wait for data transmission
        mp.WaitTime(0.1)
        
        # print(mp.get_program_rapid())
        log_results = self.client.execute_motion_program(mp)
        log_results_str = log_results.decode('ascii')
        return log_results_str

    def exec_motions_multimove(self,breakpoints,primitives1,primitives2,points1,points2,curve_js1,curve_js2,speed1,speed2,zone1,zone2):
        mp1 = MotionProgram(tool=self.tool1)
        mp2 = MotionProgram(tool=self.tool2)
        
        for i in range(len(primitives1)):
            ###force last point to be fine
            # if i==len(primitives1)-1:
            #     zone1=fine
            motion = primitives1[i]
            if motion == 'movel_fit':
                robt = self.moveL_target(self.robot1,curve_js1[breakpoints[i]],points1[i])
                mp1.MoveL(robt,speed1,zone1)

            elif motion == 'movec_fit':
                robt1, robt2 = self.moveC_target(self.robot1,curve_js1[breakpoints[i-1]],curve_js1[breakpoints[i]],points1[i][0],points1[i][1])
                mp1.MoveC(robt1,robt2,speed1,zone1)

            else: # movej_fit
                jointt = self.moveJ_target(points1[i])
                if i==0:
                    mp1.MoveAbsJ(jointt,v500,fine)
                    mp1.WaitTime(1)
                    mp1.MoveAbsJ(jointt,v500,fine)
                    mp1.WaitTime(0.1)
                else:
                    mp1.MoveAbsJ(jointt,speed1,zone1)

        for i in range(len(primitives2)):
            ###force last point to be fine
            # if i==len(primitives2)-1:
            #     zone2=fine
            motion = primitives2[i]
            if motion == 'movel_fit':
                robt = self.moveL_target(self.robot2,curve_js2[breakpoints[i]],points2[i])
                mp2.MoveL(robt,speed2,zone2)

            elif motion == 'movec_fit':
                robt1, robt2 = self.moveC_target(self.robot2,curve_js2[breakpoints[i-1]],curve_js2[breakpoints[i]],points2[i][0],points2[i][1])
                mp2.MoveC(robt1,robt2,speed2,zone2)

            else: # movej_fit
                jointt = self.moveJ_target(points2[i])
                if i==0:
                    mp2.MoveAbsJ(jointt,v500,fine)
                    mp2.WaitTime(1)
                    mp2.MoveAbsJ(jointt,v500,fine)
                    mp2.WaitTime(0.1)
                else:
                    mp2.MoveAbsJ(jointt,speed2,zone2)

        ###add sleep at the end to wait for data transmission
        mp1.WaitTime(0.1)
        mp2.WaitTime(0.1)
        
        # print(mp1.get_program_rapid())
        # print(mp2.get_program_rapid())
        log_results = self.client.execute_multimove_motion_program([mp1,mp2])
        log_results_str = log_results.decode('ascii')
        return log_results_str

        
    def extract_points(self,primitive_type,points):
        if primitive_type=='movec_fit':
            endpoints=points[8:-3].split('array')
            endpoint1=endpoints[0][:-4].split(',')
            endpoint2=endpoints[1][2:].split(',')

            return list(map(float, endpoint1)),list(map(float, endpoint2))
        else:
            endpoint=points[8:-3].split(',')
            return list(map(float, endpoint))

    def exe_from_file_w_extension(self,robot,filename,filename_js,speed,zone,extension_d=100):
        data = read_csv(filename)
        breakpoints=np.array(data['breakpoints'].tolist())
        primitives=data['primitives'].tolist()
        points=data['points'].tolist()
        curve_js=read_csv(filename_js,header=None).values
        breakpoints[1:]=breakpoints[1:]-1

        #parser
        points_list=[]
        for i in range(len(breakpoints)):
            if primitives[i]=='movel_fit':
                point=extract_points(primitives[i],points[i])
                points_list.append(point)
            elif primitives[i]=='movec_fit':
                point1,point2=extract_points(primitives[i],points[i])
                points_list.append([point1,point2])
            else:
                point=extract_points(primitives[i],points[i])
                points_list.append(point)
        
        ###initial point extension
        
        pose_start=robot.fwd(curve_js[0])
        p_start=pose_start.p
        R_start=pose_start.R
        pose_end=robot.fwd(curve_js[breakpoints[1]])
        p_end=pose_end.p
        R_end=pose_end.R
        if primitives[1]=='movel_fit':
            #find new start point
            slope_p=p_end-p_start
            slope_p=slope_p/np.linalg.norm(slope_p)
            p_start_new=p_start-extension_d*slope_p        ###extend 5cm backward

            #find new start orientation
            k,theta=R2rot(R_end@R_start.T)
            theta_new=-extension_d*theta/np.linalg.norm(p_end-p_start)
            R_start_new=rot(k,theta_new)@R_start

            #solve invkin for initial point
            q_all = robot.inv(p_start_new, R_start_new)
            temp_q=q_all-curve_js[0]
            order=np.argsort(np.linalg.norm(temp_q,axis=1))
            points_list[0]=q_all[order[0]]

        elif  primitives[1]=='movec_fit':
            #define circle first
            pose_mid=robot.fwd(curve_js[int(breakpoints[1]/2)])
            p_mid=pose_mid.p
            R_mid=pose_mid.R
            center, radius=circle_from_3point(p_start,p_end,p_mid)

            #find desired rotation angle
            angle=extension_d/radius

            #find new start point
            plane_N=np.cross(p_end-center,p_start-center)
            plane_N=plane_N/np.linalg.norm(plane_N)
            R_temp=rot(plane_N,angle)
            p_start_new=center+R_temp@(p_start-center)

            #find new start orientation
            k,theta=R2rot(R_end@R_start.T)
            theta_new=-extension_d*theta/np.linalg.norm(p_end-p_start)
            R_start_new=rot(k,theta_new)@R_start

            #solve invkin for initial point
            q_all = robot.inv(p_start_new, R_start_new)
            temp_q=q_all-curve_js[0]
            order=np.argsort(np.linalg.norm(temp_q,axis=1))
            points_list[0]=q_all[order[0]]

        else:
            #find new start point
            J_start=robot.jacobian(curve_js[0])
            qdot=curve_js[breakpoints[1]]-curve_js[0]
            v=J_start[3:,:]@qdot
            t=extension_d/v
            points_list[0]=curve_js[0]+qdot*t

        ###end point extension
        pose_start=robot.fwd(curve_js[breakpoints[-2]])
        p_start=pose_start.p
        R_start=pose_start.R
        pose_end=robot.fwd(curve_js[-1])
        p_end=pose_end.p
        R_end=pose_end.R

        if primitives[-1]=='movel_fit':
            #find new end point
            slope_p=p_end-p_start
            slope_p=slope_p/np.linalg.norm(slope_p)
            p_end_new=p_end+extension_d*slope_p        ###extend 5cm backward

            #find new end orientation
            k,theta=R2rot(R_end@R_start.T)
            theta_new=extension_d*theta/np.linalg.norm(p_end-p_start)
            R_end_new=rot(k,theta_new)@R_end

            #solve invkin for end point
            q_all = robot.inv(p_end_new, R_end_new)
            temp_q=q_all-curve_js[-1]
            order=np.argsort(np.linalg.norm(temp_q,axis=1))
            curve_js[-1]=q_all[order[0]]
            points_list[-1]=p_end_new

        elif  primitives[1]=='movec_fit':
            #define circle first
            pose_mid=robot.fwd(curve_js[int((breakpoints[-1]+breakpoints[-2])/2)])
            p_mid=pose_mid.p
            R_mid=pose_mid.R
            center, radius=circle_from_3point(p_start,p_end,p_mid)

            #find desired rotation angle
            angle=extension_d/radius

            #find new end point
            plane_N=np.cross(p_start-center,p_end-center)
            plane_N=plane_N/np.linalg.norm(plane_N)
            R_temp=rot(plane_N,angle)
            p_end_new=center+R_temp@(p_end-center)

            #find new end orientation
            k,theta=R2rot(R_end@R_start.T)
            theta_new=extension_d*theta/np.linalg.norm(p_end-p_start)
            R_end_new=rot(k,theta_new)@R_end

            #solve invkin for end point
            q_all = robot.inv(p_end_new, R_end_new)
            temp_q=q_all-curve_js[-1]
            order=np.argsort(np.linalg.norm(temp_q,axis=1))
            curve_js[-1]=q_all[order[0]]
            points_list[-1][-1]=p_end_new   #midpoint not changed

        else:
            #find new end point
            J_end=robot.jacobian(curve_js[-1])
            qdot=curve_js[-1]-curve_js[breakpoints[-2]]
            v=J_end[3:,:]@qdot
            t=extension_d/v
            points_list[-1]=curve_js[-1]+qdot*t

        return self.exec_motions(robot,primitives,breakpoints,points_list,curve_js,speed,zone)

    def exe_from_file(self,robot,filename,filename_js,speed,zone):
        data = read_csv(filename)
        breakpoints=np.array(data['breakpoints'].tolist())
        primitives=data['primitives'].tolist()
        points=data['points'].tolist()
        
        breakpoints[1:]=breakpoints[1:]-1
        

        points_list=[]
        for i in range(len(breakpoints)):
            if primitives[i]=='movel_fit':
                point=extract_points(primitives[i],points[i])
                points_list.append(point)
            elif primitives[i]=='movec_fit':
                point1,point2=extract_points(primitives[i],points[i])
                points_list.append([point1,point2])
            else:
                point=extract_points(primitives[i],points[i])
                points_list.append(point)

        curve_js=read_csv(filename_js,header=None).values
        return self.exec_motions(robot,primitives,breakpoints,points_list,curve_js,speed,zone)

    def exe_from_file_multimove(self,filename1,filename2,filename_js1,filename_js2,speed1,speed2,zone1,zone2):
        data1 = read_csv(filename1)
        breakpoints=np.array(data1['breakpoints'].tolist())
        primitives1=data1['primitives'].tolist()
        points1=data1['points'].tolist()

        data2 = read_csv(filename2)
        primitives2=data2['primitives'].tolist()
        points2=data2['points'].tolist()
        
        breakpoints[1:]=breakpoints[1:]-1
        

        points_list1=[]
        for i in range(len(breakpoints)):
            if primitives1[i]=='movel_fit':
                point=extract_points(primitives1[i],points1[i])
                points_list1.append(point)
            elif primitives1[i]=='movec_fit':
                point1,point2=extract_points(primitives1[i],points1[i])
                points_list1.append([point1,point2])
            else:
                point=extract_points(primitives1[i],points1[i])
                points_list1.append(point)

        points_list2=[]
        for i in range(len(breakpoints)):
            if primitives2[i]=='movel_fit':
                point=extract_points(primitives2[i],points2[i])
                points_list2.append(point)
            elif primitives2[i]=='movec_fit':
                point1,point2=extract_points(primitives2[i],points2[i])
                points_list2.append([point1,point2])
            else:
                point=extract_points(primitives2[i],points2[i])
                points_list2.append(point)



        curve_js1=read_csv(filename_js1,header=None).values
        curve_js2=read_csv(filename_js2,header=None).values
        return self.exec_motions_multimove(breakpoints,primitives1,primitives2,points_list1,points_list2,curve_js1,curve_js2,speed1,speed2,zone1,zone2)


    def logged_data_analysis(self,robot,df,filt=False,realrobot=False):
        q1=df[' J1'].tolist()
        q2=df[' J2'].tolist()
        q3=df[' J3'].tolist()
        q4=df[' J4'].tolist()
        q5=df[' J5'].tolist()
        q6=df[' J6'].tolist()
        cmd_num=np.array(df[' cmd_num'].tolist()).astype(float)
        #find closest to 5 cmd_num
        idx = np.absolute(cmd_num-5).argmin()
        # print('cmd_num ',cmd_num[idx])
        start_idx=np.where(cmd_num==cmd_num[idx])[0][0]
        curve_exe_js=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)[start_idx:])
        timestamp=np.array(df['timestamp'].tolist()[start_idx:]).astype(float)
        timestep=np.average(timestamp[1:]-timestamp[:-1])

        if filt:
            timestamp, curve_exe_js=lfilter(timestamp, curve_exe_js)


        act_speed=[]
        lam=[0]
        curve_exe=[]
        curve_exe_R=[]
        for i in range(len(curve_exe_js)):
            robot_pose=robot.fwd(curve_exe_js[i])
            curve_exe.append(robot_pose.p)
            curve_exe_R.append(robot_pose.R)
            if i>0:
                lam.append(lam[-1]+np.linalg.norm(curve_exe[i]-curve_exe[i-1]))
            try:
                if timestamp[i-1]!=timestamp[i] and np.linalg.norm(curve_exe_js[i-1]-curve_exe_js[i])!=0:
                    if not realrobot:
                        act_speed.append(np.linalg.norm(curve_exe[-1]-curve_exe[-2])/timestep)
                    else:
                        act_speed.append(np.linalg.norm(curve_exe[-1]-curve_exe[-2])/(timestamp[i]-timestamp[i-1]))
                else:
                    act_speed.append(act_speed[-1])      
            except IndexError:
                pass

        return lam, np.array(curve_exe), np.array(curve_exe_R),curve_exe_js, act_speed, timestamp

def main():
    ms = MotionSend()
    # data_dir="../greedy_fitting/greedy_output/"
    data_dir="../greedy_fitting/greedy_dual_output/"
    # speed={"v50":v50,"v500":v500,"v5000":v5000}
    # zone={"fine":fine,"z1":z1,"z10":z10}
    vmax = speeddata(10000,9999999,9999999,999999)
    v559 = speeddata(559,9999999,9999999,999999)
    speed={"v50":v50}#,"v500":v500,"v300":v300,"v100":v100}
    zone={"z10":z10}

    for s in speed:
        for z in zone: 
            curve_exe_js=ms.exe_from_file(data_dir+"command2.csv",data_dir+"curve_fit_js2.csv",speed[s],zone[z])
   

            # f = open(data_dir+"curve_exe"+"_"+s+"_"+z+".csv", "w")
            # f.write(curve_exe_js)
            # f.close()

if __name__ == "__main__":
    main()