import numpy as np
from numpy.linalg import norm
import general_robotics_toolbox as rox
from general_robotics_toolbox import general_robotics_toolbox_invkin as roxinv
from math import ceil,cos, sin
from pandas import *
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('../../toolbox')
from robots_def import *

dt = 0.004
robot = abb6640(R_tool=Ry(np.radians(90)),p_tool=np.array([0,0,0]))

def unit_vector(v):
    return v/norm(v)

class TCPCurve(object):
    def __init__(self,start_p,mid_p,end_p,z) -> None:
        super().__init__()
        
        d_start_mid = start_p-mid_p
        d_end_mid = end_p-mid_p
        n = unit_vector(np.cross(d_start_mid,d_end_mid))
        pre_tcp_limit_p = unit_vector(d_start_mid)*z+mid_p
        nxt_tcp_limit_p = unit_vector(d_end_mid)*z+mid_p

        A = np.vstack((np.vstack((d_start_mid,d_end_mid)),n))
        b = [np.dot(d_start_mid,pre_tcp_limit_p),np.dot(d_end_mid,nxt_tcp_limit_p),np.dot(n,pre_tcp_limit_p)]

        self.pre_tcp_limit_p = pre_tcp_limit_p
        self.nxt_tcp_limit_p = nxt_tcp_limit_p
        self.pr = np.dot(np.linalg.pinv(A),b)
        # print(A)
        # print(b)
        # print(self.pr)
        self.r = norm(self.pr-pre_tcp_limit_p)
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
        
        

class AnalyticalPredictor(object):
    def __init__(self,K,Tf) -> None:

        self.K=K
        self.Tf=Tf
        # self.K=1
        # self.Tf=10

    def predict(self,q_start,T_start,T_target,T_target_next,vel_tcp,z_tcp,z_ori):

        essential_p = []
        
        T_now = robot.fwd(q_start)

        l1 = norm(T_target.p-T_start.p)
        l2 = norm(T_target_next.p-T_target.p)
        
        in_zone_tcp = False
        in_zone_ori = False
        
        direction = unit_vector(T_target.p-T_start.p)
        
        # not in zone
        p_nominal = []
        p_nominal.append(np.dot(T_now.p-T_start.p,direction)*direction+T_start.p)
        th_nominal = [0]

        q_all = [q_start]
        T_all = [T_now]
        if z_tcp == 0:
            k_ang,th_total = rox.R2rot(np.matmul(T_now.R.T,T_target.R))
            # total time / dt
            # total_time = norm(T_target.p-p_nominal[-1].p)/vel_tcp
            dth_dl = th_total/norm(T_target.p-p_nominal[-1])

            for t in range(self.Tf):
                next_p = p_nominal[-1]+direction*np.min([vel_tcp*dt,norm(T_target.p-p_nominal[-1])])
                p_nominal.append(next_p)
                next_th = th_nominal[-1]+dth_dl*norm(p_nominal[-1]-p_nominal[-2])
                th_nominal.append(next_th)

                next_T = rox.Transform(np.matmul(T_now.R,rox.rot(k_ang,th_nominal[-1])),p_nominal[-1])
                T_all.append(next_T)

                q_prop_all = roxinv.robot6_sphericalwrist_invkin(robot.robot_def,next_T,q_all[-1])
                # q_prop_all = roxinv.robot6_sphericalwrist_invkin(robot.robot_def,next_T)
                if len(q_prop_all) == 0:
                    not_this_point_flag = True
                    break

                q_prop = q_prop_all[0]
                q_all.append(q_prop)
        else:
            tcp_curve = TCPCurve(T_start.p,T_target.p,T_target_next.p,z_tcp)
            essential_p.append(tcp_curve.pre_tcp_limit_p)
            essential_p.append(tcp_curve.nxt_tcp_limit_p)
            dl = vel_tcp*dt
            
            # do the position first
            second_half = False
            for t in range(self.Tf):
                d_corner = norm(T_target.p-p_nominal[-1])
                if d_corner <= z_tcp:
                    next_p,l_remain = tcp_curve.walk_len(p_nominal[-1],vel_tcp*dt)
                    if l_remain > 0: # remain l means change direction
                        direction = unit_vector(T_target_next.p-T_target.p)
                        next_p = next_p + direction*l_remain
                        z_tcp = 0 # no more zone
                        second_half = True # change target

                    p_nominal.append(next_p)
                else:
                    # if one more step to the zone
                    if z_tcp!=0 and (vel_tcp*dt > (d_corner-z_tcp)):
                        
                        next_p = p_nominal[-1]+direction*(d_corner-z_tcp)
                        l_remain = dl-(d_corner-z_tcp)
                        next_p,l_remain = tcp_curve.walk_len(next_p,l_remain,now_theta=0)

                        if l_remain > 0: # remain l means change direction
                            direction = unit_vector(T_target_next.p-T_target.p)
                            next_p = next_p + direction*l_remain
                            z_tcp = 0 # no more zone
                            second_half = True # change target

                        p_nominal.append(next_p)
                    else:
                        if second_half:
                            next_p = p_nominal[-1]+direction*np.min([vel_tcp*dt,norm(T_target_next.p-p_nominal[-1])])
                        else:
                            next_p = p_nominal[-1]+direction*vel_tcp*dt
                        p_nominal.append(next_p)
            
            # then do the orientation
            k_ang_1,th_total_1 = rox.R2rot(np.matmul(T_start.R.T,T_target.R))
            zone_start_ori = (1-z_ori/l1)*th_total_1
            dth_dl_1 = th_total_1/l1
            k_ang_2,th_total_2 = rox.R2rot(np.matmul(T_target.R.T,T_target_next.R))
            zone_end_ori = (z_ori/l2)*th_total_2
            dth_dl_2 = th_total_2/l2
            k_ang_now,th_total_now = rox.R2rot(np.matmul(T_now.R.T,T_target.R))
            dth_dl_now = th_total_now/norm(T_target.p-T_now.p)
            
            zone_start_R = np.matmul(T_start.R,rox.rot(k_ang_1,zone_start_ori))
            zone_end_R = np.matmul(T_target.R,rox.rot(k_ang_2,zone_end_ori))
            k_ang_half,th_total_half = rox.R2rot(np.matmul(zone_start_R.T,zone_end_R))
            dth_dl_half = th_total_half/(tcp_curve.r*tcp_curve.ang_total+2*(z_ori-z_tcp))

            second_half = False
            for pno_i in range(1,len(p_nominal)):
                d_corner = norm(T_target.p-p_nominal[pno_i])
                d_corner_pre = norm(T_target.p-p_nominal[pno_i-1])
                if d_corner <= z_ori:
                    if d_corner_pre > z_ori:
                        dl_after_zori = dl-(d_corner_pre-z_ori)
                        th_nominal.append(dth_dl_half*dl_after_zori)
                        this_R = np.matmul(zone_start_R,rox.rot(k_ang_half,th_nominal[-1]))
                    else:
                        th_nominal.append(dth_dl_half*dl)
                        this_R = np.matmul(zone_start_R,rox.rot(k_ang_half,th_nominal[-1]))
                else:
                    if d_corner_pre <= z_ori:
                        dl_after_zori = (d_corner-z_ori)
                        th_nominal.append(dth_dl_2*dl_after_zori)
                        this_R = np.matmul(T_target.R,rox.rot(k_ang_2,th_nominal[-1]))
                        z_ori=0 # no more z ori
                        second_half = True
                    else:
                        if second_half:
                            th_nominal.append(dth_dl_2*dl)
                            this_R = np.matmul(T_target.R,rox.rot(k_ang_2,th_nominal[-1]))
                        else:
                            th_nominal.append(dth_dl_now*dl)
                            this_R = np.matmul(T_now.R,rox.rot(k_ang_now,th_nominal[-1]))

                T_all.append(rox.Transform(this_R,p_nominal[pno_i]))
                q_prop_all = roxinv.robot6_sphericalwrist_invkin(robot.robot_def,T_all[-1],q_all[-1])
                if len(q_prop_all) == 0:
                    not_this_point_flag = True
                    break
                q_prop = q_prop_all[0]
                q_all.append(q_prop)

        # remove the first (current) q
        q_all.pop(0)
        return q_all,essential_p

def draw_frame(ax,all_T,color=None):

    ratio = 5
    for T in all_T:
        if color is None:
            ax.plot3D([T.p[0],T.p[0]+ratio*T.R[0,0]], [T.p[1],T.p[1]+ratio*T.R[1,0]], [T.p[2],T.p[2]+ratio*T.R[2,0]], 'red')
            ax.plot3D([T.p[0],T.p[0]+ratio*T.R[0,1]], [T.p[1],T.p[1]+ratio*T.R[1,1]], [T.p[2],T.p[2]+ratio*T.R[2,1]], 'green')
            ax.plot3D([T.p[0],T.p[0]+ratio*T.R[0,2]], [T.p[1],T.p[1]+ratio*T.R[1,2]], [T.p[2],T.p[2]+ratio*T.R[2,2]], 'blue')
        else:
            ax.plot3D([T.p[0],T.p[0]+ratio*T.R[0,0]], [T.p[1],T.p[1]+ratio*T.R[1,0]], [T.p[2],T.p[2]+ratio*T.R[2,0]], color)
            ax.plot3D([T.p[0],T.p[0]+ratio*T.R[0,1]], [T.p[1],T.p[1]+ratio*T.R[1,1]], [T.p[2],T.p[2]+ratio*T.R[2,1]], color)
            ax.plot3D([T.p[0],T.p[0]+ratio*T.R[0,2]], [T.p[1],T.p[1]+ratio*T.R[1,2]], [T.p[2],T.p[2]+ratio*T.R[2,2]], color)

def main():
    
    # load motion instructions
    col_names=['motion','p1_x','p1_y','p1_z','p1_qw','p1_qx','p1_qy','p1_qz', \
                'p1_cf1','p1_cf2','p1_cf3','p1_cf4',\
                'p2_x','p2_y','p2_z','p2_qw','p2_qx','p2_qy','p2_qz', \
                'p2_cf1','p2_cf2','p2_cf3','p2_cf4',\
                'p3_x','p3_y','p3_z','p3_qw','p3_qx','p3_qy','p3_qz', \
                'p3_cf1','p3_cf2','p3_cf3','p3_cf4',\
                'vel','zone'] 
    motions = read_csv("data/0_input.txt", names=col_names)
    p1 = np.array([motions['p1_x'].tolist(),motions['p1_y'].tolist(),motions['p1_z'].tolist()])
    R1 = rox.q2R(np.array([motions['p1_qw'].tolist(),motions['p1_qx'].tolist(),motions['p1_qy'].tolist(),motions['p1_qz'].tolist()]))
    T1 = rox.Transform(R1,p1)
    p2 = np.array([motions['p2_x'].tolist(),motions['p2_y'].tolist(),motions['p2_z'].tolist()])
    R2 = rox.q2R(np.array([motions['p2_qw'].tolist(),motions['p2_qx'].tolist(),motions['p2_qy'].tolist(),motions['p2_qz'].tolist()]))
    T2 = rox.Transform(R2,p2)
    p3 = np.array([motions['p3_x'].tolist(),motions['p3_y'].tolist(),motions['p3_z'].tolist()])
    R3 = rox.q2R(np.array([motions['p3_qw'].tolist(),motions['p3_qx'].tolist(),motions['p3_qy'].tolist(),motions['p3_qz'].tolist()]))
    T3 = rox.Transform(R3,p3)
    print(T1)

    vel_profile = motions['vel'].tolist()[0]
    zone_profile = motions['zone'].tolist()[0]

    # load logged joints
    col_names=['timestamp','cmd_num','q1', 'q2', 'q3','q4', 'q5', 'q6'] 
    data = read_csv("data/0_gt.csv", names=col_names,skiprows = 1)
    curve_q1=data['q1'].tolist()
    curve_q2=data['q2'].tolist()
    curve_q3=data['q3'].tolist()
    curve_q4=data['q4'].tolist()
    curve_q5=data['q5'].tolist()
    curve_q6=data['q6'].tolist()
    joint_angles=np.deg2rad(np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T)

    curve_x=np.array([])
    curve_y=np.array([])
    curve_z=np.array([])
    curve_T = []
    for i in range(len(joint_angles)):
        T = robot.fwd(joint_angles[i])
        curve_x = np.append(curve_x,T.p[0])
        curve_y = np.append(curve_y,T.p[1])
        curve_z = np.append(curve_z,T.p[2])
        curve_T.append(T)

    plt.figure()
    # ax = plt.axes(projection='3d')
    fig, ax = plt.subplots()
    ax= plt.axes(projection='3d')

    Tf = 20
    ap = AnalyticalPredictor(1,Tf)
    global second_half
    second_half = False

    def animate(an_i):
        global second_half

        ax.clear()

        q_start_i = an_i

        if second_half:
            q_next,essential_p = ap.predict(joint_angles[q_start_i],T2,T3,T3,vel_profile,0,15)
        else:
            q_next,essential_p = ap.predict(joint_angles[q_start_i],T1,T2,T3,vel_profile,zone_profile,15)
        
        essential_p = np.array(essential_p)
        q_next.insert(0,joint_angles[q_start_i])

        curve_pre_x=np.array([])
        curve_pre_y=np.array([])
        curve_pre_z=np.array([])
        curve_pre_T = []
        for i in range(len(q_next)):
            T = robot.fwd(q_next[i])
            curve_pre_x = np.append(curve_pre_x,T.p[0])
            curve_pre_y = np.append(curve_pre_y,T.p[1])
            curve_pre_z = np.append(curve_pre_z,T.p[2])
            curve_pre_T.append(T)
        
        # curve_x=np.array([])
        # curve_y=np.array([])
        # curve_z=np.array([])
        # for i in range(len(joint_angles)):
        #     T = robot.fwd(joint_angles[i])
        #     curve_x = np.append(curve_x,T.p[0])
        #     curve_y = np.append(curve_y,T.p[1])
        #     curve_z = np.append(curve_z,T.p[2])

        if not second_half:
            this_p = np.array([curve_x[an_i],curve_y[an_i],curve_z[an_i]])
            next_p = np.array([curve_x[an_i+1],curve_y[an_i+1],curve_z[an_i+1]])
            dthis_t2 = norm(this_p-T2.p)
            dnext_t2 = norm(next_p-T2.p)

            # if this p in zone but next dont
            # else if within distance, next p is farther than this p
            if (dthis_t2<=zone_profile) :
                if (dnext_t2>zone_profile):
                    second_half = True
            elif (dthis_t2<=10) and ((dthis_t2<=dnext_t2)):
                second_half = True
            else:
                pass

        
        
        ax.plot3D(curve_x, curve_y, curve_z, 'gray')
        ax.plot3D(curve_pre_x, curve_pre_y, curve_pre_z, 'red')
        ax.scatter3D([p1[0],p2[0],p3[0]],[p1[1],p2[1],p3[1]],[p1[2],p2[2],p3[2]])

        if len(essential_p) > 0:
            ax.scatter3D(essential_p[:,0],essential_p[:,1],essential_p[:,2],'magenta')
        # draw_frame(ax,[T1,T2,T3],'blue')
        # draw_frame(ax,curve_pre_T[0::10],'red')
        # draw_frame(ax,curve_T[0::10],'green')
    ani = FuncAnimation(fig, animate, frames=459, interval=20, repeat=False)
    plt.show()

    # q_next = np.array(q_next)
    # for i in range(6):
    #     plt.figure()
    #     ax = plt.axes()
    #     ax.plot(joint_angles[:,i],'green')
    #     ax.plot(q_next[:,i],'red')
    #     plt.show()

if __name__ == '__main__':
    main()