from turtle import shape
import numpy as np
from numpy.linalg import norm
from math import pi, cos, sin, radians, sqrt
import time
from pandas import *
from qpsolvers import solve_qp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import general_robotics_toolbox as rox
from general_robotics_toolbox import general_robotics_toolbox_invkin as roxinv

import sys
sys.path.append('../../toolbox')
from robots_def import *

dt = 0.004
robot = abb6640(R_tool=Ry(np.radians(90)),p_tool=np.array([0,0,0]))

def unit_vector(a,b):
    return (b-a)/norm(b-a)

class ZoneFit(object):
    def __init__(self,joint_angles,move_start,move_mid,move_end,zone) -> None:
        
        # define traj
        self.joint_angles = joint_angles
        self.move_start = np.array(move_start)
        self.move_mid = np.array(move_mid)
        self.move_end = np.array(move_end)
        self.zone_start_p = unit_vector(self.move_mid,self.move_start)*zone+self.move_mid
        self.zone_end_p = unit_vector(self.move_mid,self.move_end)*zone+self.move_mid

        self.zone_start_i = None
        self.zone_end_i = None

        self.traj_x=np.array([])
        self.traj_y=np.array([])
        self.traj_z=np.array([])
        self.traj_T = []
        for i in range(len(joint_angles)):
            T = robot.fwd(joint_angles[i])
            self.traj_x = np.append(self.traj_x,T.p[0])
            self.traj_y = np.append(self.traj_y,T.p[1])
            self.traj_z = np.append(self.traj_z,T.p[2])
            self.traj_T.append(T)

            if np.linalg.norm(T.p-move_mid) <= zone and (self.zone_start_i is None):
                self.zone_start_i = i
            if np.linalg.norm(T.p-move_mid) > zone and (self.zone_start_i is not None) and (self.zone_end_i is None):
                self.zone_end_i = i
        
        # transformation for fitting curves
        x_axis = unit_vector(self.zone_start_p,self.zone_end_p)
        z_axis = np.cross(x_axis,unit_vector(self.zone_start_p,self.move_mid))
        z_axis = z_axis/norm(z_axis)
        y_axis = np.cross(z_axis,x_axis)
        T_fit = rox.Transform(np.array([x_axis,y_axis,z_axis]).T,self.zone_start_p)
        self.T_fit = T_fit.inv()

        # transformation for L's
        y_axis = unit_vector(self.move_start,self.move_mid)
        z_axis = np.cross(unit_vector(self.move_mid,self.move_end),y_axis)
        z_axis = z_axis/norm(z_axis)
        x_axis = np.cross(y_axis,z_axis)
        T_fit_L1 = rox.Transform(np.array([x_axis,y_axis,z_axis]).T,self.move_start)
        self.T_fit_L1 = T_fit_L1.inv()

        y_axis = unit_vector(self.move_end,self.move_mid)
        z_axis = np.cross(unit_vector(self.move_start,self.move_mid),y_axis)
        z_axis = z_axis/norm(z_axis)
        x_axis = np.cross(y_axis,z_axis)
        T_fit_L2 = rox.Transform(np.array([x_axis,y_axis,z_axis]).T,self.move_end)
        self.T_fit_L2 = T_fit_L2.inv()

    def L_fit_loss(self):
        
        x_st,y_st,z_st=self.move_start
        x_md,y_md,z_md=self.move_mid
        x_ed,y_ed,z_ed=self.move_end

        d_start_mid = norm(self.move_start-self.move_mid)
        d_start_mid_2d = norm(self.move_start[:2]-self.move_mid[:2])
        d_mid_end = norm(self.move_mid-self.move_end)
        d_mid_end_2d = norm(self.move_mid[:2]-self.move_end[:2])


        # loss (distance) of the first moveL
        distance_1_3d = np.sqrt(((self.traj_x[:self.zone_start_i]-x_st)**2+(self.traj_y[:self.zone_start_i]-y_st)**2+(self.traj_z[:self.zone_start_i]-z_st)**2)\
                        -(np.dot(np.array([self.traj_x[:self.zone_start_i]-x_st,self.traj_y[:self.zone_start_i]-y_st,self.traj_z[:self.zone_start_i]-z_st]).T,[x_md-x_st,y_md-y_st,z_md-z_st])/d_start_mid)**2)
        distance_1_2d = np.sqrt(((self.traj_x[:self.zone_start_i]-x_st)**2+(self.traj_y[:self.zone_start_i]-y_st)**2)\
                        -(np.dot(np.array([self.traj_x[:self.zone_start_i]-x_st,self.traj_y[:self.zone_start_i]-y_st]).T,[x_md-x_st,y_md-y_st])/d_start_mid_2d)**2)
        # loss (distance) of the second moveL
        distance_2_3d = np.sqrt(((self.traj_x[self.zone_end_i:]-x_md)**2+(self.traj_y[self.zone_end_i:]-y_md)**2+(self.traj_z[self.zone_end_i:]-z_md)**2)\
                        -(np.dot(np.array([self.traj_x[self.zone_end_i:]-x_md,self.traj_y[self.zone_end_i:]-y_md,self.traj_z[self.zone_end_i:]-z_md]).T,[x_ed-x_md,y_ed-y_md,z_ed-z_md])/d_mid_end)**2)
        distance_2_2d = np.sqrt(((self.traj_x[self.zone_end_i:]-x_md)**2+(self.traj_y[self.zone_end_i:]-y_md)**2)\
                        -(np.dot(np.array([self.traj_x[self.zone_end_i:]-x_md,self.traj_y[self.zone_end_i:]-y_md]).T,[x_ed-x_md,y_ed-y_md])/d_mid_end_2d)**2)
        
        # print(distance_1_3d)
        # print(distance_1_2d)
        # print(distance_2_3d)
        # print(distance_2_2d)

        return distance_1_3d,distance_1_2d,distance_2_3d,distance_2_2d

    def parabola_fit(self):

        # fine the trans of the zone curve for fit
        # so that zone start at 0,0
        # transform the entire thing s.t. zone start p at 0,0,0, zone end-zone start is x axis, z axis is the plane norm
        # It's self.T_fit
        T_fit = self.T_fit

        # transform all the data points
        traj_x_zone = self.traj_x[self.zone_start_i:self.zone_end_i]
        traj_y_zone = self.traj_y[self.zone_start_i:self.zone_end_i]
        traj_z_zone = self.traj_z[self.zone_start_i:self.zone_end_i]
        traj_x_zone_t,traj_y_zone_t,traj_z_zone_t = \
            np.add(np.matmul(T_fit.R,np.array([traj_x_zone,traj_y_zone,traj_z_zone])).T,T_fit.p).T
        x_zed_t,y_zed_t,z_zed_t = np.matmul(T_fit.R,self.zone_end_p)+T_fit.p
        x_md_t,y_md_t,z_mid_t = np.matmul(T_fit.R,self.move_mid)+T_fit.p

        # find the parabola (there's no DoF in finding the parabola)
        # [x^2 x 1][a;b;c]=[y]
        # [2x 1 0][a;b;c]=[slope]
        A=np.array([[0,0,1],[x_zed_t**2,x_zed_t,1],[0,1,0]])
        b=np.array([0,y_zed_t,y_md_t/x_md_t])
        alpha,beta,gamma = np.matmul(np.linalg.pinv(A),b)

        # calculate y least square loss (fitting loss)
        y_loss = np.sqrt(np.average((np.polyval([alpha,beta,gamma],traj_x_zone_t)-traj_y_zone_t)**2))

        # drawing and transform back
        draw_x = np.linspace(0,x_zed_t,101)
        draw_y = np.polyval([alpha,beta,gamma],draw_x)
        T_fit_inv = T_fit.inv()
        draw_x,draw_y,draw_z = np.add(np.matmul(T_fit_inv.R,np.array([draw_x,draw_y,np.zeros(len(draw_x))])).T,T_fit_inv.p).T

        return [alpha,beta,gamma],y_loss,draw_x,draw_y,draw_z

    def cubic_fit(self):

        # fine the trans of the zone curve for fit
        # so that zone start at 0,0
        # transform the entire thing s.t. zone start p at 0,0,0, zone end-zone start is x axis, z axis is the plane norm
        # It's self.T_fit
        T_fit = self.T_fit

        # transform all the data points
        traj_x_zone = self.traj_x[self.zone_start_i:self.zone_end_i]
        traj_y_zone = self.traj_y[self.zone_start_i:self.zone_end_i]
        traj_z_zone = self.traj_z[self.zone_start_i:self.zone_end_i]
        traj_x_zone_t,traj_y_zone_t,traj_z_zone_t = \
            np.add(np.matmul(T_fit.R,np.array([traj_x_zone,traj_y_zone,traj_z_zone])).T,T_fit.p).T
        x_zed_t,y_zed_t,z_zed_t = np.matmul(T_fit.R,self.zone_end_p)+T_fit.p
        x_md_t,y_md_t,z_mid_t = np.matmul(T_fit.R,self.move_mid)+T_fit.p

        # find the Cubic (there's no DoF in finding the Cubic)
        # [x^3 x^2 x 1][a;b;c;d]=[y]
        # [3x^2 2x 1 0][a;b;c;d]=[slope]
        A=np.array([[0,0,0,1],[x_zed_t**3,x_zed_t**2,x_zed_t,1],[0,0,1,0],[3*x_zed_t**2,2*x_zed_t,1,0]])
        b=np.array([0,y_zed_t,y_md_t/x_md_t,(y_zed_t-y_md_t)/(x_zed_t-x_md_t)])
        alpha,beta,gamma,delta = np.matmul(np.linalg.pinv(A),b)

        # print(alpha,beta,gamma,delta)
        # print(np.matmul(A,[alpha,beta,gamma]))

        # test qp
        P = np.array([traj_x_zone_t**3,traj_x_zone_t**2,traj_x_zone_t**1,np.ones(len(traj_x_zone_t))])
        q = -np.matmul(traj_y_zone_t,P.T)
        P = np.matmul(P,P.T)
        # print(solve_qp(P,q,None,None,A,b))

        # calculate y least square loss (fitting loss)
        y_loss = np.sqrt(np.average((np.polyval([alpha,beta,gamma,delta],traj_x_zone_t)-traj_y_zone_t)**2))
        # print(y_loss)

        # drawing and transform back
        draw_x = np.linspace(0,x_zed_t,101)
        draw_y = np.polyval([alpha,beta,gamma,delta],draw_x)
        T_fit_inv = T_fit.inv()
        draw_x,draw_y,draw_z = np.add(np.matmul(T_fit_inv.R,np.array([draw_x,draw_y,np.zeros(len(draw_x))])).T,T_fit_inv.p).T

        return [alpha,beta,gamma,delta],y_loss,draw_x,draw_y,draw_z

    def quintic_fit(self):

        # fine the trans of the zone curve for fit
        # so that zone start at 0,0
        # transform the entire thing s.t. zone start p at 0,0,0, zone end-zone start is x axis, z axis is the plane norm
        # It's self.T_fit
        T_fit = self.T_fit

        # transform all the data points
        traj_x_zone = self.traj_x[self.zone_start_i:self.zone_end_i]
        traj_y_zone = self.traj_y[self.zone_start_i:self.zone_end_i]
        traj_z_zone = self.traj_z[self.zone_start_i:self.zone_end_i]
        traj_x_zone_t,traj_y_zone_t,traj_z_zone_t = \
            np.add(np.matmul(T_fit.R,np.array([traj_x_zone,traj_y_zone,traj_z_zone])).T,T_fit.p).T
        x_zed_t,y_zed_t,z_zed_t = np.matmul(T_fit.R,self.zone_end_p)+T_fit.p
        x_md_t,y_md_t,z_mid_t = np.matmul(T_fit.R,self.move_mid)+T_fit.p

        # find the quintic (there's 2 DoF in finding the quintic)
        # use qp
        A=np.array([[0,0,0,0,0,1],[x_zed_t**5,x_zed_t**4,x_zed_t**3,x_zed_t**2,x_zed_t,1],\
                    [0,0,0,0,1,0],[5*x_zed_t**4,4*x_zed_t**3,3*x_zed_t**2,2*x_zed_t,1,0]])
        b=np.array([0,y_zed_t,y_md_t/x_md_t,(y_zed_t-y_md_t)/(x_zed_t-x_md_t)])
        P = np.array([traj_x_zone_t**5,traj_x_zone_t**4,traj_x_zone_t**3,traj_x_zone_t**2,\
            traj_x_zone_t**1,np.ones(len(traj_x_zone_t))])
        q = -np.matmul(traj_y_zone_t,P.T)
        P = np.matmul(P,P.T)
        param = solve_qp(P,q,None,None,A,b)
        # print(param)

        # calculate y least square loss (fitting loss)
        y_loss = np.sqrt(np.average((np.polyval(param,traj_x_zone_t)-traj_y_zone_t)**2))
        # print(y_loss)

        # drawing and transform back
        draw_x = np.linspace(0,x_zed_t,101)
        draw_y = np.polyval(param,draw_x)
        T_fit_inv = T_fit.inv()
        draw_x,draw_y,draw_z = np.add(np.matmul(T_fit_inv.R,np.array([draw_x,draw_y,np.zeros(len(draw_x))])).T,T_fit_inv.p).T

        return param,y_loss,draw_x,draw_y,draw_z

def draw(start_p,mid_p,end_p,zone_fit,zone_x,zone_y,zone_z,save_folder):
    
    zone_start,zone_end=zone_fit.zone_start_p,zone_fit.zone_end_p
    zone_start_i,zone_end_i = zone_fit.zone_start_i,zone_fit.zone_end_i
    traj_x,traj_y,traj_z = zone_fit.traj_x,zone_fit.traj_y,zone_fit.traj_z
    traj_x_zone = zone_fit.traj_x[zone_start_i:zone_end_i]
    traj_y_zone = zone_fit.traj_y[zone_start_i:zone_end_i]
    traj_z_zone = zone_fit.traj_z[zone_start_i:zone_end_i]

    # transform all the data points
    T_fit = zone_fit.T_fit
    traj_x_zone_t,traj_y_zone_t,traj_z_zone_t = \
        np.add(np.matmul(T_fit.R,np.array([traj_x_zone,traj_y_zone,traj_z_zone])).T,T_fit.p).T
    traj_x_t,traj_y_t,traj_z_t = \
        np.add(np.matmul(T_fit.R,np.array([traj_x,traj_y,traj_z])).T,T_fit.p).T
    zone_x_t,zone_y_t,zone_z_t = \
        np.add(np.matmul(T_fit.R,np.array([zone_x,zone_y,zone_z])).T,T_fit.p).T
    zone_start_t = np.matmul(T_fit.R,zone_start)+T_fit.p
    zone_end_t = np.matmul(T_fit.R,zone_end)+T_fit.p
    start_p_t = np.matmul(T_fit.R,start_p)+T_fit.p
    mid_p_t = np.matmul(T_fit.R,mid_p)+T_fit.p
    end_p_t = np.matmul(T_fit.R,end_p)+T_fit.p

    
    # 1. 3d pics
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot3D([start_p[0],zone_start[0]], [start_p[1],zone_start[1]], [start_p[2],zone_start[2]], 'gray')
    ax.plot3D([end_p[0],zone_end[0]], [end_p[1],zone_end[1]], [end_p[2],zone_end[2]], 'gray')
    ax.plot3D(zone_x,zone_y,zone_z,'gray')
    ax.scatter3D(traj_x[:zone_start_i],traj_y[:zone_start_i],traj_z[:zone_start_i],s=1,c='tab:pink')
    ax.scatter3D(traj_x[zone_end_i:],traj_y[zone_end_i:],traj_z[zone_end_i:],s=1,c='tab:pink')
    ax.scatter3D(traj_x_zone,traj_y_zone,traj_z_zone,s=1,c='tab:blue')
    ax.scatter3D(start_p[0],start_p[1],start_p[2],c='red')
    ax.scatter3D(mid_p[0],mid_p[1],mid_p[2],c='orange')
    ax.scatter3D(end_p[0],end_p[1],end_p[2],c='green')
    ax.view_init(35,125)
    ax.set_xlabel('x-axis (mm)')
    ax.set_ylabel('y-axis (mm)')
    ax.set_zlabel('z-axis (mm)')
    fig.savefig(save_folder+'3d.png')
    # plt.show()
    # 2. zone 3d
    plt.close(fig)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax= plt.axes(projection='3d')
    ax.plot3D(zone_x,zone_y,zone_z,'gray')
    ax.scatter3D(traj_x_zone,traj_y_zone,traj_z_zone,s=1)
    ax.scatter3D(zone_start[0],zone_start[1],zone_start[2],c='red')
    ax.scatter3D(mid_p[0],mid_p[1],mid_p[2],c='orange')
    ax.scatter3D(zone_end[0],zone_end[1],zone_end[2],c='green')
    ax.view_init(25,135)
    ax.set_xlabel('x-axis (mm)')
    ax.set_ylabel('y-axis (mm)')
    ax.set_zlabel('z-axis (mm)')
    fig.savefig(save_folder+'3d_zone.png')
    # plt.show()

    

    # 3. zone 2d xy
    plt.close(fig)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(zone_x_t,zone_y_t,'gray')
    ax.scatter(traj_x_zone_t,traj_y_zone_t,s=1)
    ax.scatter(zone_start_t[0],zone_start_t[1],c='red')
    ax.scatter(mid_p_t[0],mid_p_t[1],c='orange')
    ax.scatter(zone_end_t[0],zone_end_t[1],c='green')
    ax.set_xlabel('x-axis (Transform) (mm)')
    ax.set_ylabel('y-axis (Transform) (mm)')
    fig.savefig(save_folder+'zone_xy.png')
    # plt.show()
    # 4. zone 2d yz
    plt.close(fig)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(zone_y_t,zone_z_t,'gray')
    ax.scatter(traj_y_zone_t,traj_z_zone_t,s=1)
    ax.scatter(zone_start_t[1],zone_start_t[2],c='red')
    ax.scatter(mid_p_t[1],mid_p_t[2],c='orange')
    ax.scatter(zone_end_t[1],zone_end_t[2],c='green')
    ax.set_xlabel('y-axis (Transform) (mm)')
    ax.set_ylabel('z-axis (Transform) (mm)')
    fig.savefig(save_folder+'zone_yz.png')
    # plt.show()
    # 5. zone 2d yz
    plt.close(fig)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(zone_x_t,zone_z_t,'gray')
    ax.scatter(traj_x_zone_t,traj_z_zone_t,s=1)
    ax.scatter(zone_start_t[0],zone_start_t[2],c='red')
    ax.scatter(mid_p_t[0],mid_p_t[2],c='orange')
    ax.scatter(zone_end_t[0],zone_end_t[2],c='green')
    ax.set_xlabel('x-axis (Transform) (mm)')
    ax.set_ylabel('z-axis (Transform) (mm)')
    fig.savefig(save_folder+'zone_xz.png')
    # plt.show()
    plt.close(fig)
    plt.clf()
    # 6. traj 2d xy
    plt.close(fig)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot([start_p_t[0],zone_start_t[0]], [start_p_t[1],zone_start_t[1]], 'gray')
    ax.plot([end_p_t[0],zone_end_t[0]], [end_p_t[1],zone_end_t[1]], 'gray')
    ax.plot(zone_x_t,zone_y_t,'gray')
    ax.scatter(traj_x_t,traj_y_t,s=1)
    ax.scatter(start_p_t[0],start_p_t[1],c='red')
    ax.scatter(mid_p_t[0],mid_p_t[1],c='orange')
    ax.scatter(end_p_t[0],end_p_t[1],c='green')
    ax.set_xlabel('x-axis (Transform) (mm)')
    ax.set_ylabel('y-axis (Transform) (mm)')
    fig.savefig(save_folder+'traj_xy.png')
    # plt.show()
    # 7. traj 2d yz
    plt.close(fig)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot([start_p_t[1],zone_start_t[1]], [start_p_t[2],zone_start_t[2]], 'gray')
    ax.plot([end_p_t[1],zone_end_t[1]], [end_p_t[2],zone_end_t[2]], 'gray')
    ax.plot(zone_y_t,zone_z_t,'gray')
    ax.scatter(traj_y_t,traj_z_t,s=1)
    ax.scatter(start_p_t[1],start_p_t[2],c='red')
    ax.scatter(mid_p_t[1],mid_p_t[2],c='orange')
    ax.scatter(end_p_t[1],end_p_t[2],c='green')
    ax.set_xlabel('y-axis (Transform) (mm)')
    ax.set_ylabel('z-axis (Transform) (mm)')
    fig.savefig(save_folder+'traj_yz.png')
    # plt.show()
    # 8. traj 2d yz
    plt.close(fig)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot([start_p_t[0],zone_start_t[0]], [start_p_t[2],zone_start_t[2]], 'gray')
    ax.plot([end_p_t[0],zone_end_t[0]], [end_p_t[2],zone_end_t[2]], 'gray')
    ax.plot(zone_x_t,zone_z_t,'gray')
    ax.scatter(traj_x_t,traj_z_t,s=1)
    ax.scatter(start_p_t[0],start_p_t[2],c='red')
    ax.scatter(mid_p_t[0],mid_p_t[2],c='orange')
    ax.scatter(end_p_t[0],end_p_t[2],c='green')
    ax.set_xlabel('x-axis (Transform) (mm)')
    ax.set_ylabel('z-axis (Transform) (mm)')
    fig.savefig(save_folder+'traj_xz.png')
    # plt.show()
    plt.close(fig)
    plt.clf()
    

def draw_L(start_p,mid_p,end_p,zone_fit,zone_x,zone_y,zone_z,save_folder):

    zone_start,zone_end=zone_fit.zone_start_p,zone_fit.zone_end_p
    zone_start_i,zone_end_i = zone_fit.zone_start_i,zone_fit.zone_end_i

    T_fit = zone_fit.T_fit_L1
    traj_x_L1 = zone_fit.traj_x[:zone_start_i]
    traj_y_L1 = zone_fit.traj_y[:zone_start_i]
    traj_z_L1 = zone_fit.traj_z[:zone_start_i]
    traj_x_L1_t,traj_y_L1_t,traj_z_L1_t = \
        np.add(np.matmul(T_fit.R,np.array([traj_x_L1,traj_y_L1,traj_z_L1])).T,T_fit.p).T
    zone_start_t = np.matmul(T_fit.R,zone_start)+T_fit.p
    start_p_t = np.matmul(T_fit.R,start_p)+T_fit.p
    mid_p_t = np.matmul(T_fit.R,mid_p)+T_fit.p

    # 9. L1 2d xy
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot([start_p_t[0],zone_start_t[0]], [start_p_t[1],zone_start_t[1]], 'gray')
    ax.scatter(traj_x_L1_t,traj_y_L1_t,s=1)
    ax.scatter(start_p_t[0],start_p_t[1],c='red')
    ax.scatter(mid_p_t[0],mid_p_t[1],c='orange')
    ax.set_xlabel('x-axis (Transform) (mm)')
    ax.set_ylabel('y-axis (Transform) (mm)')
    fig.savefig(save_folder+'L1_xy.png')

    T_fit = zone_fit.T_fit_L2
    traj_x_L2 = zone_fit.traj_x[zone_end_i:]
    traj_y_L2 = zone_fit.traj_y[zone_end_i:]
    traj_z_L2 = zone_fit.traj_z[zone_end_i:]
    traj_x_L2_t,traj_y_L2_t,traj_z_L2_t = \
        np.add(np.matmul(T_fit.R,np.array([traj_x_L2,traj_y_L2,traj_z_L2])).T,T_fit.p).T
    zone_end_t = np.matmul(T_fit.R,zone_end)+T_fit.p
    end_p_t = np.matmul(T_fit.R,end_p)+T_fit.p
    mid_p_t = np.matmul(T_fit.R,mid_p)+T_fit.p

    # 10. L2 2d xy
    plt.close(fig)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot([end_p_t[0],zone_end_t[0]], [end_p_t[1],zone_end_t[1]], 'gray')
    ax.scatter(traj_x_L2_t,traj_y_L2_t,s=1)
    ax.scatter(end_p_t[0],end_p_t[1],c='green')
    ax.scatter(mid_p_t[0],mid_p_t[1],c='orange')
    ax.set_xlabel('x-axis (Transform) (mm)')
    ax.set_ylabel('y-axis (Transform) (mm)')
    fig.savefig(save_folder+'L2_xy.png')

def draw_all_curve(zone_fit,para_x,para_y,para_z,cubi_x,cubi_y,cubi_z,quin_x,quin_y,quin_z,save_folder):
    
    zone_start,zone_end=zone_fit.zone_start_p,zone_fit.zone_end_p
    zone_start_i,zone_end_i = zone_fit.zone_start_i,zone_fit.zone_end_i
    traj_x_zone = zone_fit.traj_x[zone_start_i:zone_end_i]
    traj_y_zone = zone_fit.traj_y[zone_start_i:zone_end_i]
    traj_z_zone = zone_fit.traj_z[zone_start_i:zone_end_i]

    # transform all the data points
    T_fit = zone_fit.T_fit
    traj_x_zone_t,traj_y_zone_t,traj_z_zone_t = \
        np.add(np.matmul(T_fit.R,np.array([traj_x_zone,traj_y_zone,traj_z_zone])).T,T_fit.p).T
    para_x_t,para_y_t,para_z_t = \
        np.add(np.matmul(T_fit.R,np.array([para_x,para_y,para_z])).T,T_fit.p).T
    cubi_x_t,cubi_y_t,cubi_z_t = \
        np.add(np.matmul(T_fit.R,np.array([cubi_x,cubi_y,cubi_z])).T,T_fit.p).T
    quin_x_t,quin_y_t,quin_z_t = \
        np.add(np.matmul(T_fit.R,np.array([quin_x,quin_y,quin_z])).T,T_fit.p).T
    zone_start_t = np.matmul(T_fit.R,zone_start)+T_fit.p
    zone_end_t = np.matmul(T_fit.R,zone_end)+T_fit.p
    mid_p_t = np.matmul(T_fit.R,zone_fit.move_mid)+T_fit.p

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(zone_start_t[0],zone_start_t[1],c='red')
    ax.scatter(mid_p_t[0],mid_p_t[1],c='orange')
    ax.scatter(zone_end_t[0],zone_end_t[1],c='green')
    ax.scatter(traj_x_zone_t,traj_y_zone_t,s=1)
    ax.plot(para_x_t,para_y_t,'y')
    ax.plot(cubi_x_t,cubi_y_t,'c')
    ax.plot(quin_x_t,quin_y_t,'m')
    ax.set_xlabel('x-axis (mm)')
    ax.set_ylabel('y-axis (mm)')
    fig.savefig(save_folder+'zone_curve_compare.png')
    plt.close(fig)
    plt.clf()    

def draw_joints(zone_fit,joint_angles,save_folder):
    
    joint_angles = np.rad2deg(joint_angles.T)

    # 1. joint position
    for ji in range(6):
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot()
        draw_target = joint_angles[ji]
        ax.scatter(np.arange(0,len(joint_angles[ji])),draw_target,s=1)
        min_draw = np.min(draw_target)
        max_draw = np.max(draw_target)
        ax.plot([zone_fit.zone_start_i,zone_fit.zone_start_i], [min_draw,max_draw], 'gray')
        ax.plot([zone_fit.zone_end_i,zone_fit.zone_end_i], [min_draw,max_draw], 'gray')
        ax.set_xlabel('time index')
        ax.set_ylabel('joint (rad)')
        fig.savefig(save_folder+'_J'+str(ji+1)+'_joint_position.png')
        plt.close(fig)

    # 2. joint velocity
    for ji in range(6):
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot()
        draw_target = np.append(np.diff(joint_angles[ji])/dt,0)
        ax.scatter(np.arange(0,len(joint_angles[ji])),draw_target,s=1)
        min_draw = np.min(draw_target)
        max_draw = np.max(draw_target)
        ax.plot([zone_fit.zone_start_i,zone_fit.zone_start_i], [min_draw,max_draw], 'gray')
        ax.plot([zone_fit.zone_end_i,zone_fit.zone_end_i], [min_draw,max_draw], 'gray')
        ax.set_xlabel('time index')
        ax.set_ylabel('joint (rad)')
        fig.savefig(save_folder+'_J'+str(ji+1)+'_joint_velocity.png')
        plt.close(fig)

    # 3. joint acceleration
    for ji in range(6):
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot()
        draw_target = np.append(np.diff(joint_angles[ji],n=2)/(dt**2),[0,0])
        ax.scatter(np.arange(0,len(joint_angles[ji])),draw_target,s=1)
        min_draw = np.min(draw_target)
        max_draw = np.max(draw_target)
        ax.plot([zone_fit.zone_start_i,zone_fit.zone_start_i], [min_draw,max_draw], 'gray')
        ax.plot([zone_fit.zone_end_i,zone_fit.zone_end_i], [min_draw,max_draw], 'gray')
        ax.set_xlabel('time index')
        ax.set_ylabel('joint (rad)')
        fig.savefig(save_folder+'_J'+str(ji+1)+'_joint_acceleration.png')
        plt.close(fig)

def main():
    
    # folder to read
    data_folder = 'data_param/'
    # data_folder = 'data_param_vertical/'

    # data info
    start_p = np.array([2300,1000,600])
    end_p = np.array([1300,-1000,600])
    all_Rq = rox.R2q(rox.rot([0,1,0],pi/2))
    side_l = 200
    zone = 10
    
    x_divided = 11
    step_x = (end_p[0]-start_p[0])/(x_divided-1)
    y_divided = 11
    step_y = (end_p[1]-start_p[1])/(y_divided-1)
    angels = [90,120,150]

    vel=50

    # result data
    y_loss_zone_cubic = {} 
    y_loss_zone_cubic[angels[0]] = np.zeros((x_divided,y_divided))
    y_loss_zone_cubic[angels[1]] = np.zeros((x_divided,y_divided))
    y_loss_zone_cubic[angels[2]] = np.zeros((x_divided,y_divided))
    y_loss_zone_quintic = {}
    y_loss_zone_quintic[angels[0]] = np.zeros((x_divided,y_divided))
    y_loss_zone_quintic[angels[1]] = np.zeros((x_divided,y_divided))
    y_loss_zone_quintic[angels[2]] = np.zeros((x_divided,y_divided))
    z_height_diff_ave = {}
    z_height_diff_ave[angels[0]] = np.zeros((x_divided,y_divided))
    z_height_diff_ave[angels[1]] = np.zeros((x_divided,y_divided))
    z_height_diff_ave[angels[2]] = np.zeros((x_divided,y_divided))
    z_height_diff_std = {}
    z_height_diff_std[angels[0]] = np.zeros((x_divided,y_divided))
    z_height_diff_std[angels[1]] = np.zeros((x_divided,y_divided))
    z_height_diff_std[angels[2]] = np.zeros((x_divided,y_divided))
    zone_z_height_diff_ave = {}
    zone_z_height_diff_ave[angels[0]] = np.zeros((x_divided,y_divided))
    zone_z_height_diff_ave[angels[1]] = np.zeros((x_divided,y_divided))
    zone_z_height_diff_ave[angels[2]] = np.zeros((x_divided,y_divided))
    zone_z_height_diff_std = {}
    zone_z_height_diff_std[angels[0]] = np.zeros((x_divided,y_divided))
    zone_z_height_diff_std[angels[1]] = np.zeros((x_divided,y_divided))
    zone_z_height_diff_std[angels[2]] = np.zeros((x_divided,y_divided))
    moveL_1_diff_ave = {}
    moveL_1_diff_ave[angels[0]] = np.zeros((x_divided,y_divided))
    moveL_1_diff_ave[angels[1]] = np.zeros((x_divided,y_divided))
    moveL_1_diff_ave[angels[2]] = np.zeros((x_divided,y_divided))
    moveL_2_diff_ave = {}
    moveL_2_diff_ave[angels[0]] = np.zeros((x_divided,y_divided))
    moveL_2_diff_ave[angels[1]] = np.zeros((x_divided,y_divided))
    moveL_2_diff_ave[angels[2]] = np.zeros((x_divided,y_divided))

    param_quintic = []
    for i in range(6):
        param_quintic_ = {}
        param_quintic_[angels[0]] = np.zeros((x_divided,y_divided))
        param_quintic_[angels[1]] = np.zeros((x_divided,y_divided))
        param_quintic_[angels[2]] = np.zeros((x_divided,y_divided))
        param_quintic.append(param_quintic_)


    # where to start
    start_xi = 0
    start_yi = 0

    for pxi in range(start_xi,x_divided):
        start_xi = 0
        for pyi in range(start_yi,y_divided):
            start_yi = 0
            for ang in angels:
                # load logged joints
                col_names=['timestamp','cmd_num','q1', 'q2', 'q3','q4', 'q5', 'q6'] 
                data = read_csv(data_folder+"log_"+str(vel)+"_"+"{:02d}".format(pxi)+"_"+"{:02d}".format(pyi)+"_"+\
                            str(ang)+".csv", names=col_names,skiprows = 1)
                curve_q1=data['q1'].tolist()
                curve_q2=data['q2'].tolist()
                curve_q3=data['q3'].tolist()
                curve_q4=data['q4'].tolist()
                curve_q5=data['q5'].tolist()
                curve_q6=data['q6'].tolist()
                joint_angles=np.deg2rad(np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T)

                px = start_p[0]+step_x*pxi
                py = start_p[1]+step_y*pyi
                # move_start = [px-side_l*cos(radians(ang/2)),py+side_l*sin(radians(ang/2)),start_p[2]]
                # move_end = [px-side_l*cos(radians(ang/2)),py-side_l*sin(radians(ang/2)),start_p[2]]
                move_start = [px,py+side_l*sin(radians(ang/2)),start_p[2]-side_l*cos(radians(ang/2))]
                move_end = [px,py-side_l*sin(radians(ang/2)),start_p[2]-side_l*cos(radians(ang/2))]
                move_mid = [px,py,start_p[2]]

                zone_fit = ZoneFit(joint_angles,move_start,move_mid,move_end,zone)

                print("Line fit")
                d1_3d,d1_2d,d2_3d,d2_2d = zone_fit.L_fit_loss()
                print("Parabola fit")
                param_para,loss_para,draw_x_para,draw_y_para,draw_z_para = zone_fit.parabola_fit()
                save_folder=data_folder+'result/'+"log_"+str(vel)+"_"+"{:02d}".format(pxi)+"_"+"{:02d}".format(pyi)+"_"+\
                            str(ang)+"_para_"
                # draw(move_start,move_mid,move_end,zone_fit,draw_x_para,draw_y_para,draw_z_para,save_folder)
                print("Cubic fit")
                param_cubi,loss_cubi,draw_x_cubi,draw_y_cubi,draw_z_cubi = zone_fit.cubic_fit()
                save_folder=data_folder+'result/'+"log_"+str(vel)+"_"+"{:02d}".format(pxi)+"_"+"{:02d}".format(pyi)+"_"+\
                            str(ang)+"_cubi_"
                draw(move_start,move_mid,move_end,zone_fit,draw_x_cubi,draw_y_cubi,draw_z_cubi,save_folder)
                print("Quintic fit")
                param_quin,loss_quin,draw_x_quin,draw_y_quin,draw_z_quin = zone_fit.quintic_fit()
                save_folder=data_folder+'result/'+"log_"+str(vel)+"_"+"{:02d}".format(pxi)+"_"+"{:02d}".format(pyi)+"_"+\
                            str(ang)+"_quin_"
                draw(move_start,move_mid,move_end,zone_fit,draw_x_quin,draw_y_quin,draw_z_quin,save_folder)
                
                # draw compare
                save_folder=data_folder+'result/'+"log_"+str(vel)+"_"+"{:02d}".format(pxi)+"_"+"{:02d}".format(pyi)+"_"+\
                            str(ang)+"_"
                draw_L(move_start,move_mid,move_end,zone_fit,draw_x_quin,draw_y_quin,draw_z_quin,save_folder)
                draw_all_curve(zone_fit,draw_x_para,draw_y_para,draw_z_para,\
                                        draw_x_cubi,draw_y_cubi,draw_z_cubi,\
                                        draw_x_quin,draw_y_quin,draw_z_quin,save_folder)

                # draw joint position, velocity, acceleration
                # draw_joints(zone_fit,joint_angles,save_folder)

                
                # transform all the data points
                traj_x_zone = zone_fit.traj_x[zone_fit.zone_start_i:zone_fit.zone_end_i]
                traj_y_zone = zone_fit.traj_y[zone_fit.zone_start_i:zone_fit.zone_end_i]
                traj_z_zone = zone_fit.traj_z[zone_fit.zone_start_i:zone_fit.zone_end_i]
                T_fit = zone_fit.T_fit
                traj_x_zone_t,traj_y_zone_t,traj_z_zone_t = \
                    np.add(np.matmul(T_fit.R,np.array([traj_x_zone,traj_y_zone,traj_z_zone])).T,T_fit.p).T
                traj_x_t,traj_y_t,traj_z_t = \
                    np.add(np.matmul(T_fit.R,np.array([zone_fit.traj_x,zone_fit.traj_y,zone_fit.traj_z])).T,T_fit.p).T
                move_start_t = np.matmul(T_fit.R,zone_fit.move_start)+T_fit.p

                y_loss_zone_cubic[ang][pxi,pyi] = loss_cubi
                y_loss_zone_quintic[ang][pxi,pyi] = loss_quin
                z_height_diff_ave[ang][pxi,pyi] = np.mean(traj_z_t-move_start_t[2])
                z_height_diff_std[ang][pxi,pyi] = np.std(traj_z_t-move_start_t[2])
                zone_z_height_diff_ave[ang][pxi,pyi] = np.mean(traj_z_zone_t-move_start_t[2])
                zone_z_height_diff_std[ang][pxi,pyi] = np.std(traj_z_zone_t-move_start_t[2])
                moveL_1_diff_ave[ang][pxi,pyi] = np.mean(d1_3d)
                moveL_2_diff_ave[ang][pxi,pyi] = np.mean(d2_3d)

                for param_i in range(6):
                    param_quintic[param_i][ang][pxi,pyi] = param_quin[param_i]

                # for testing
                # break
                ############
            print("Progress:",(pxi*y_divided+pyi+1)/(x_divided*y_divided)*100,"%")
            # for testing
            # break
            ############
        # for testing
        # break
        ############
    
    for ang in angels:
        plt.clf()
        plt.imshow(y_loss_zone_cubic[ang], cmap='viridis')
        plt.colorbar()
        save_folder=data_folder+'result/y_loss_zone_cubic'+'_'+str(ang)+"_"
        plt.savefig(save_folder+'.png')
        plt.clf()
        plt.imshow(y_loss_zone_quintic[ang], cmap='viridis')
        plt.colorbar()
        save_folder=data_folder+'result/y_loss_zone_quintic'+'_'+str(ang)+"_"
        plt.savefig(save_folder+'.png')
        plt.clf()
        plt.imshow(z_height_diff_ave[ang], cmap='viridis')
        plt.colorbar()
        save_folder=data_folder+'result/z_height_diff_ave'+'_'+str(ang)+"_"
        plt.savefig(save_folder+'.png')
        plt.clf()
        plt.imshow(z_height_diff_std[ang], cmap='viridis')
        plt.colorbar()
        save_folder=data_folder+'result/z_height_diff_std'+'_'+str(ang)+"_"
        plt.savefig(save_folder+'.png')
        plt.clf()
        plt.imshow(zone_z_height_diff_ave[ang], cmap='viridis')
        plt.colorbar()
        save_folder=data_folder+'result/zone_z_height_diff_ave'+'_'+str(ang)+"_"
        plt.savefig(save_folder+'.png')
        plt.clf()
        plt.imshow(zone_z_height_diff_std[ang], cmap='viridis')
        plt.colorbar()
        save_folder=data_folder+'result/zone_z_height_diff_std'+'_'+str(ang)+"_"
        plt.savefig(save_folder+'.png')
        plt.clf()
        plt.imshow(moveL_1_diff_ave[ang], cmap='viridis')
        plt.colorbar()
        save_folder=data_folder+'result/moveL_1_diff_ave'+'_'+str(ang)+"_"
        plt.savefig(save_folder+'.png')
        plt.clf()
        plt.imshow(moveL_2_diff_ave[ang], cmap='viridis')
        plt.colorbar()
        save_folder=data_folder+'result/moveL_2_diff_ave'+'_'+str(ang)+"_"
        plt.savefig(save_folder+'.png')
        plt.clf()

        for param_i in range(6):
            plt.imshow(param_quintic[param_i][ang], cmap='viridis')
            plt.colorbar()
            save_folder=data_folder+'result/param_quintic_'+str(5-param_i)+'ord_'+str(ang)+"_"
            plt.savefig(save_folder+'.png')
            plt.clf()

        # plt.show()

if __name__ == '__main__':
    main()