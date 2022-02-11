########
# This module utilized https://github.com/johnwason/abb_motion_program_exec
# and send whatever the motion primitives that algorithms generate
# to RobotStudio
########

import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
import csv
sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../toolbox')
from robots_def import *
# from robot_def import *

robot = abb6640()

def quadrant(q):
    temp=np.ceil(np.array([q[0],q[3],q[5]])/(np.pi/2))-1
    
    if q[4] < 0:
        last = 1
    else:
        last = 0

    return np.hstack((temp,[last])).astype(int)

class MotionSend(object):
    def __init__(self) -> None:
        
        self.client = MotionProgramExecClient()

        # paint gun with dummy load data
        # to get the real load data, we need to know the mass 
        # # and the inertia of the paint gun

        quatR = R2q(rot([0,1,0],math.radians(30)))
        self.tool = tooldata(True,pose([50,0,450],[quatR[0],quatR[1],quatR[2],quatR[3]]),loaddata(0.001,[0,0,0.001],[1,0,0,0],0,0,0))

    def moveL_target(self,q,point):
        quat=R2q(robot.fwd(q).R)
        cf=quadrant(q)

        robt = robtarget([point[0], point[1], point[2]], [ quat[0], quat[1], quat[2], quat[3]], confdata(cf[0],cf[1],cf[2],cf[3]),[9E+09]*6)
        return robt
    
    def moveC_target(self,q1,q2,point1,point2):

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

    def exec_motions(self,primitives,breakpoints,points,curve_backproj_js_filename='Curve_backproj_js',vel_data=5000,zone_data=0,\
                    save_file=False,save_file_name='log.csv',show_rapid=False,show_result=False):

        vel = v5000
        if vel_data == 50:
            vel=v50
        elif vel_data == 500:
            vel=v500
        else:
            vel=v5000
        
        zone=fine
        if zone_data == 0:
            zone=fine
        elif zone_data == 1:
            zone=z1
        else:
            zone=z10

        col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
        data = read_csv("../data/from_ge/"+curve_backproj_js_filename+".csv", names=col_names)
        curve_q1=data['q1'].tolist()
        curve_q2=data['q2'].tolist()
        curve_q3=data['q3'].tolist()
        curve_q4=data['q4'].tolist()
        curve_q5=data['q5'].tolist()
        curve_q6=data['q6'].tolist()
        curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

        mp = MotionProgram(tool=self.tool)
        
        for i in range(len(primitives)):
            motion = primitives[i]

            # Motion program should start and end with fine zone
            if i == len(primitives)-1 or i==0:
                this_zone = fine
            else:
                this_zone = zone

            if motion == 'movel_fit':
                robt = self.moveL_target(curve_js[breakpoints[i]],points[i][0])
                mp.MoveL(robt,vel,this_zone)

            elif motion == 'movec_fit':
                robt1, robt2 = self.moveC_target(curve_js[breakpoints[i-1]],curve_js[breakpoints[i]],points[i][0],points[i][1])
                mp.MoveC(robt1,robt2,vel,this_zone)

            else: # movej_fit
                jointt = self.moveJ_target(points[i][0])
                mp.MoveAbsJ(jointt,vel,this_zone)
        
        if show_rapid:
            print(mp.get_program_rapid())

        log_results = self.client.execute_motion_program(mp)
        if save_file:
            with open(save_file_name,"wb") as f:
                f.write(log_results)

        log_results_str = log_results.decode('ascii')
        if show_result:
            print(log_results_str)

        log_results_dict = {}
        rows = log_results_str.split("\r\n")
        for row in rows[:-1]:
            if len(log_results_dict) == 0:
                log_results_dict['timestamp']=[]
                log_results_dict['cmd_num']=[]
                log_results_dict['joint_angle']=[]
                continue
            col = row.split(", ")
            log_results_dict['timestamp'].append(float(col[0]))
            log_results_dict['cmd_num'].append(float(col[1]))
            log_results_dict['joint_angle'].append(np.array([float(col[2]),float(col[3]),float(col[4]),float(col[5]),float(col[6]),float(col[7])]))

        return log_results_dict
    
    def exec_motions_from_file(self,data_file_name='command_backproj.csv',vel_data=5000,zone_data=0,\
                    save_file=False,save_file_name='log.csv',show_rapid=False,show_result=False):
        
        data = read_csv(data_file_name)
        breakpoints=data['breakpoints'].tolist()
        primitives=data['primitives'].tolist()
        points=data['points'].tolist()

        points_list=[]
        for i in range(len(breakpoints)):
            if primitives[i]=='movel_fit':
                point=self.extract_points(primitives[i],points[i])
                points_list.append(point)
            elif primitives[i]=='movec_fit':
                point1,point2=self.extract_points(primitives[i],points[i])
                points_list.append([point1,point2])
            else:
                point=self.extract_points(primitives[i],points[i])
                points_list.append(point)
        
        self.exec_motions(primitives,breakpoints,points_list,vel_data=vel_data,zone_data=zone_data,\
            save_file=save_file,save_file_name=save_file_name,show_rapid=show_rapid,show_result=show_result)

    def extract_points(self,primitive_type,points):
        if primitive_type=='movec_fit':
            endpoints=points[8:-3].split('array')
            endpoint1=endpoints[0][:-4].split(',')
            endpoint2=endpoints[1][2:].split(',')

            return list(map(float, endpoint1)),list(map(float, endpoint2))
        else:
            endpoint=points[8:-3].split(',')
            return [list(map(float, endpoint))]

def main():

    ms = MotionSend()
    # ms.exec_motions(primitives,breakpoints,points_list)
    ms.exec_motions_from_file(data_file_name='command_backproj.csv',save_file=True,save_file_name='logged_joints/log.csv',show_result=True)
    # ms.exec_motions(primitives,breakpoints,points_list,vel_data=50,save_file=True,save_file_name='logged_joints/log_v50.csv')
    # ms.exec_motions_from_file(data_file_name='command_backproj.csv',zone_data=1,save_file=True,save_file_name='logged_joints/log_z1.csv')
    # ms.exec_motions(primitives,breakpoints,points_list,zone_data=1)
    # ms.exec_motions(primitives,breakpoints,points_list,zone_data=10,save_file=True,save_file_name='logged_joints/log_z10.csv')



if __name__ == "__main__":
    main()
