from cmath import cos, sin
from matplotlib.transforms import Transform
import numpy as np
import general_robotics_toolbox as rox
from general_robotics_toolbox import general_robotics_toolbox_invkin as roxinv
from math import ceil

import sys
sys.path.append('../../toolbox')
from robots_def import *

dt = 0.004
robot = abb6640(R_tool=Ry(np.radians(90)),p_tool=np.array([0,0,0]))

def unit_vector(v):
    return v/np.linalg.norm(v)

class AnalyticalPredictor(object):
    def __init__(self,K,Tf) -> None:

        self.K=K
        self.Tf=Tf
        self.K=1
        self.Tf=10

    def predict(self,q_start,T_start,T_target,T_target_next,vel_tcp,vel_ori,z_tcp,z_ori):
        
        T_now = robot.fwd(q_start)
        
        in_zone_tcp = False
        in_zone_ori = False
        
        direction = unit_vector(T_target.p-T_start.p)
        
        # not in zone
        p_nominal = []
        p_nominal.append(np.dot(T_now.p-T_start.p,direction)*direction+T_start.p)
        th_nominal = [0]
        k_ang,th_total = rox.R2rot(T_now.R.T*T_target.R)
        
        # total time / dt
        total_stamps = ceil(np.linalg.norm(T_target.p-p_nominal[-1].p)/vel_tcp/dt)
        dth = th_total/total_stamps

        q_all = [q_start]
        T_all = [rox.Transform(rox.rot(k_ang,th_nominal[-1]),p_nominal[-1])]
        if z_tcp == 0:
            for t in range(self.Tf):
                next_p = p_nominal[-1]+direction*np.min(vel_tcp*dt,np.linalg.norm(T_target-p_nominal[-1]))
                p_nominal.append(next_p)
                next_th = th_nominal[-1]+dth
                th_nominal.append(next_th)

                next_T = rox.Transform(rox.rot(k_ang,th_nominal[-1]),p_nominal[-1])
                T_all.append(next_T)

                q_prop_all = roxinv.robot6_sphericalwrist_invkin(robot.robot_def,next_T,q_all[-1])
                if len(q_prop_all) == 0:
                    not_this_point_flag = True
                    break
                q_prop = q_prop_all[0]
                q_all.append[q_prop]
        else:
            tcp_curve = TCPCurve(T_start.p,T_target.p,T_target_next.p)
            
            # do the position first
            for t in range(self.Tf):
                d_corner = np.linalg.norm(T_target.p-p_nominal[-1])
                if d_corner <= z_tcp:
                    next_p,l_remain = tcp_curve.walk_len(p_nominal[-1],vel_tcp*dt)
                    if l_remain > 0: # remain l means change direction
                        direction = unit_vector(T_target_next-T_target)
                        next_p = next_p + direction*l_remain
                        z_tcp = 0 # no more zone

                    p_nominal.append(next_p)
                else:
                    # if one more step to the zone
                    if z_tcp!=0 and (vel_tcp*dt > (d_corner-z_tcp)):
                        total_walk = vel_tcp*dt
                        next_p = p_nominal[-1]+direction*(d_corner-z_tcp)
                        next_p,l_remain = tcp_curve.walk_len(next_p,l_remain,now_theta=0)

                        if l_remain > 0: # remain l means change direction
                            direction = unit_vector(T_target_next-T_target)
                            next_p = next_p + direction*l_remain
                            z_tcp = 0 # no more zone

                        p_nominal.append(next_p)
                    else:
                        next_p = p_nominal[-1]+direction*vel_tcp*dt
                        p_nominal.append(next_p)
            
            # then do the orientation
                

        # remove the first (current) q
        q_all.pop(0)
        return q_all

            
class TCPCurve(object):
    def __init__(self,start_p,mid_p,end_p,z) -> None:
        super().__init__()
        
        d_start_mid = start_p-mid_p
        d_end_mid = end_p-mid_p
        n = unit_vector(np.cross(d_start_mid,d_end_mid))
        pre_tcp_limit_p = unit_vector(d_start_mid)*z+mid_p
        nxt_tcp_limit_p = unit_vector(d_end_mid)*z+mid_p

        A = np.vstack(np.vstack((d_start_mid,d_end_mid)),n)
        b = np.dot(d_start_mid,pre_tcp_limit_p)+np.dot(d_end_mid,nxt_tcp_limit_p)+np.dot(n,pre_tcp_limit_p)

        self.pre_tcp_limit_p = pre_tcp_limit_p
        self.nxt_tcp_limit_p = nxt_tcp_limit_p
        self.pr = np.linalg.pinv(A)*b
        self.r = np.linalg.norm(self.pr-pre_tcp_limit_p)
        self.ang_total = np.arccos(np.dot(unit_vector(pre_tcp_limit_p-self.pr),unit_vector(nxt_tcp_limit_p-self.pr)))

        self.a = (pre_tcp_limit_p-self.pr)/self.r
        self.b = (nxt_tcp_limit_p-self.pr-self.r*cos(self.ang_total)*self.a)/(self.r*sin(self.ang_total))


    def walk_len(self,start_p,l,now_theta=None):
        
        dtheta = l/self.r

        if now_theta is None:
            now_theta = np.arccos(np.dot(unit_vector(start_p-self.pr),unit_vector(self.pre_tcp_limit_p-self.pr)))

        if now_theta+dtheta > self.ang_total:
            actual_l = (self.ang_total-now_theta)*self.r
            return self.nxt_tcp_limit_p,l-actual_l
        else:
            p_des = self.pr+self.r*cos(now_theta+dtheta)*self.a+self.r*sin(now_theta+dtheta)*self.b
            return p_des,0
