import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from abb_motion_program_exec_client import *
from robots_def import *
from error_check import *


class MotionSend(object):
    def __init__(self) -> None:
        
        self.client = MotionProgramExecClient()
        # self.robot=abb6640()
        # quatR = R2q(rot([0,1,0],math.radians(30)))
        # self.tool = tooldata(True,pose([50,0,450],[quatR[0],quatR[1],quatR[2],quatR[3]]),loaddata(1,[0,0,0.001],[1,0,0,0],0,0,0))

        ###with fake link
        self.robot=abb6640(d=50)
        quatR = R2q(rot([0,1,0],math.radians(30)))
        self.tool = tooldata(True,pose([75,0,493.30127019],[quatR[0],quatR[1],quatR[2],quatR[3]]),loaddata(1,[0,0,0.001],[1,0,0,0],0,0,0))

    def moveL_target(self,q,point):
        quat=R2q(self.robot.fwd(q).R)
        cf=quadrant(q)
        robt = robtarget([point[0], point[1], point[2]], [ quat[0], quat[1], quat[2], quat[3]], confdata(cf[0],cf[1],cf[2],cf[3]),[9E+09]*6)
        return robt
    
    def moveC_target(self,q1,q2,point1,point2):
        quat1=R2q(self.robot.fwd(q1).R)
        cf1=quadrant(q1)
        quat2=R2q(self.robot.fwd(q2).R)
        cf2=quadrant(q2)
        robt1 = robtarget([point1[0], point1[1], point1[2]], [ quat1[0], quat1[1], quat1[2], quat1[3]], confdata(cf1[0],cf1[1],cf1[2],cf1[3]),[0]*6)
        robt2 = robtarget([point2[0], point2[1], point2[2]], [ quat2[0], quat2[1], quat2[2], quat2[3]], confdata(cf2[0],cf2[1],cf2[2],cf2[3]),[0]*6)
        return robt1, robt2

    def moveJ_target(self,q):
        q = np.rad2deg(q)
        jointt = jointtarget([q[0],q[1],q[2],q[3],q[4],q[5]],[0]*6)
        return jointt

    def exec_motions(self,primitives,breakpoints,points,curve_js,speed,zone):
        mp = MotionProgram(tool=self.tool)
        
        for i in range(len(primitives)):
            ###force last point to be fine
            if i==len(primitives)-1:
                zone=fine
            motion = primitives[i]
            if motion == 'movel_fit':
                robt = self.moveL_target(curve_js[breakpoints[i]],points[i][0])
                mp.MoveL(robt,speed,zone)

            elif motion == 'movec_fit':
                robt1, robt2 = self.moveC_target(curve_js[breakpoints[i-1]],curve_js[breakpoints[i]],points[i][0],points[i][1])
                mp.MoveC(robt1,robt2,speed,zone)

            else: # movej_fit
                jointt = self.moveJ_target(points[i][0])
                if i==0:
                    mp.MoveAbsJ(jointt,v500,fine)
                    mp.WaitTime(1)
                    mp.MoveAbsJ(jointt,v500,fine)
                    mp.WaitTime(0.1)
                else:
                    mp.MoveAbsJ(jointt,speed,zone)
        ###add sleep at the end to wait for data transmission
        mp.WaitTime(0.1)
        
        print(mp.get_program_rapid())
        log_results = self.client.execute_motion_program(mp)
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

    def exe_from_file(self,ms,filename,filename_js,speed,zone):
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
        curve_exe_js=ms.exec_motions(primitives,breakpoints,points_list,curve_js,speed,zone)
        return curve_exe_js

    def logged_data_analysis(self,df):
        q1=df[' J1'].tolist()
        q2=df[' J2'].tolist()
        q3=df[' J3'].tolist()
        q4=df[' J4'].tolist()
        q5=df[' J5'].tolist()
        q6=df[' J6'].tolist()
        cmd_num=np.array(df[' cmd_num'].tolist()).astype(float)
        start_idx=np.where(cmd_num==6)[0][0]
        curve_exe_js=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)[start_idx:])
        timestamp=np.array(df['timestamp'].tolist()[start_idx:]).astype(float)

        act_speed=[]
        lam=[0]
        curve_exe=[]
        curve_exe_R=[]
        for i in range(len(curve_exe_js)):
            robot_pose=self.robot.fwd(curve_exe_js[i])
            curve_exe.append(robot_pose.p)
            curve_exe_R.append(robot_pose.R)
            if i>0:
                lam.append(lam[-1]+np.linalg.norm(curve_exe[i]-curve_exe[i-1]))
            try:
                if timestamp[i-1]!=timestamp[i] and np.linalg.norm(curve_exe_js[i-1]-curve_exe_js[i])!=0:
                    act_speed.append(np.linalg.norm(curve_exe[-1]-curve_exe[-2])/(timestamp[i]-timestamp[i-1]))
                else:
                    act_speed.append(act_speed[-1])      
            except IndexError:
                pass

        return lam, np.array(curve_exe), np.array(curve_exe_R), act_speed, timestamp

def main():
    ms = MotionSend()
    data_dir="fitting_output_new/threshold0.5/"
    # data_dir="fitting_output_new/Jon/"
    # speed={"v50":v50,"v500":v500,"v5000":v5000}
    # zone={"fine":fine,"z1":z1,"z10":z10}
    vmax = speeddata(10000,9999999,9999999,999999)
    v559 = speeddata(559,9999999,9999999,999999)
    speed={"v50":v50}#,"v500":v500,"v300":v300,"v100":v100}
    zone={"z10":z10}

    for s in speed:
        for z in zone: 
            curve_exe_js=ms.exe_from_file(ms,data_dir+"command.csv",data_dir+"curve_fit_js.csv",speed[s],zone[z])
   

            f = open(data_dir+"curve_exe"+"_"+s+"_"+z+".csv", "w")
            f.write(curve_exe_js)
            f.close()

if __name__ == "__main__":
    main()