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

from abb_motion_program_exec_client import *

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
        
        x_st,y_st,z_st=self.move_start
        x_md,y_md,z_md=self.move_mid
        x_ed,y_ed,z_ed=self.move_end
        x_zst,y_zst,z_zst=self.zone_start_p
        x_zed,y_zed,z_zed=self.zone_end_p

        traj_x_zone = self.traj_x[self.zone_start_i:self.zone_end_i]
        traj_y_zone = self.traj_y[self.zone_start_i:self.zone_end_i]
        
        # fine the trans of the zone curve for fit
        # so that zone start at 0,0
        th = pi/2
        trans_2d = np.matmul(np.array([[cos(th),-sin(th),0],[sin(th),cos(th),0],[0,0,1]]),np.array([[1,0,-x_zst],[0,1,-y_zst],[0,0,1]]))
        x_zed_t,y_zed_t,_ = np.matmul(trans_2d,[x_zed,y_zed,1])
        x_md_t,y_md_t,_ = np.matmul(trans_2d,[x_md,y_md,1])
        # print(x_zst,y_zst)
        # print(x_zed,y_zed)
        # print(x_zed_t,y_zed_t)
        # transform all the data points
        traj_x_zone_t,traj_y_zone_t,_ = np.matmul(trans_2d,np.array([traj_x_zone,traj_y_zone,np.ones(len(traj_x_zone))]))
        # print(traj_x_zone_t)
        # print(traj_y_zone_t)
        # find the parabola (there's no DoF in finding the parabola)
        # [x^2 x 1][a;b;c]=[y]
        # [2x 1 0][a;b;c]=[slope]
        A=np.array([[0,0,1],[x_zed_t**2,x_zed_t,1],[0,1,0]])
        b=np.array([0,y_zed_t,y_md_t/x_md_t])
        # print(A)
        # print(b)
        # print(np.matmul(np.linalg.pinv(A),b))
        alpha,beta,gamma = np.matmul(np.linalg.pinv(A),b)

        # print(alpha,beta,gamma)
        # print(np.matmul(A,[alpha,beta,gamma]))

        # calculate y least square loss (fitting loss)
        y_loss = np.sqrt(np.average((np.polyval([alpha,beta,gamma],traj_x_zone_t)-traj_y_zone_t)**2))
        # print(y_loss)

        draw_x = np.linspace(0,x_zed_t,101)
        draw_y = np.polyval([alpha,beta,gamma],draw_x)
        th = -pi/2
        trans_2d_inv = np.matmul(np.array([[1,0,x_zst],[0,1,y_zst],[0,0,1]]),np.array([[cos(th),-sin(th),0],[sin(th),cos(th),0],[0,0,1]]))
        draw_x,draw_y,_ = np.matmul(trans_2d_inv,np.array([draw_x,draw_y,np.ones(len(draw_x))]))

        return [alpha,beta,gamma],y_loss,draw_x,draw_y

    def cubic_fit(self):
        
        x_st,y_st,z_st=self.move_start
        x_md,y_md,z_md=self.move_mid
        x_ed,y_ed,z_ed=self.move_end
        x_zst,y_zst,z_zst=self.zone_start_p
        x_zed,y_zed,z_zed=self.zone_end_p

        traj_x_zone = self.traj_x[self.zone_start_i:self.zone_end_i]
        traj_y_zone = self.traj_y[self.zone_start_i:self.zone_end_i]
        
        # fine the trans of the zone curve for fit
        # so that zone start at 0,0
        th = pi/2
        trans_2d = np.matmul(np.array([[cos(th),-sin(th),0],[sin(th),cos(th),0],[0,0,1]]),np.array([[1,0,-x_zst],[0,1,-y_zst],[0,0,1]]))
        x_zed_t,y_zed_t,_ = np.matmul(trans_2d,[x_zed,y_zed,1])
        x_md_t,y_md_t,_ = np.matmul(trans_2d,[x_md,y_md,1])
        # transform all the data points
        traj_x_zone_t,traj_y_zone_t,_ = np.matmul(trans_2d,np.array([traj_x_zone,traj_y_zone,np.ones(len(traj_x_zone))]))
        # find the parabola (there's no DoF in finding the parabola)
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

        draw_x = np.linspace(0,x_zed_t,101)
        draw_y = np.polyval([alpha,beta,gamma,delta],draw_x)
        th = -pi/2
        trans_2d_inv = np.matmul(np.array([[1,0,x_zst],[0,1,y_zst],[0,0,1]]),np.array([[cos(th),-sin(th),0],[sin(th),cos(th),0],[0,0,1]]))
        draw_x,draw_y,_ = np.matmul(trans_2d_inv,np.array([draw_x,draw_y,np.ones(len(draw_x))]))

        return [alpha,beta,gamma,delta],y_loss,draw_x,draw_y

    def quintic_fit(self):
        
        x_st,y_st,z_st=self.move_start
        x_md,y_md,z_md=self.move_mid
        x_ed,y_ed,z_ed=self.move_end
        x_zst,y_zst,z_zst=self.zone_start_p
        x_zed,y_zed,z_zed=self.zone_end_p

        traj_x_zone = self.traj_x[self.zone_start_i:self.zone_end_i]
        traj_y_zone = self.traj_y[self.zone_start_i:self.zone_end_i]
        
        # fine the trans of the zone curve for fit
        # so that zone start at 0,0
        th = pi/2
        trans_2d = np.matmul(np.array([[cos(th),-sin(th),0],[sin(th),cos(th),0],[0,0,1]]),np.array([[1,0,-x_zst],[0,1,-y_zst],[0,0,1]]))
        x_zed_t,y_zed_t,_ = np.matmul(trans_2d,[x_zed,y_zed,1])
        x_md_t,y_md_t,_ = np.matmul(trans_2d,[x_md,y_md,1])
        # transform all the data points
        traj_x_zone_t,traj_y_zone_t,_ = np.matmul(trans_2d,np.array([traj_x_zone,traj_y_zone,np.ones(len(traj_x_zone))]))
        # find the parabola (there's no DoF in finding the parabola)
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

        draw_x = np.linspace(0,x_zed_t,101)
        draw_y = np.polyval(param,draw_x)
        th = -pi/2
        trans_2d_inv = np.matmul(np.array([[1,0,x_zst],[0,1,y_zst],[0,0,1]]),np.array([[cos(th),-sin(th),0],[sin(th),cos(th),0],[0,0,1]]))
        draw_x,draw_y,_ = np.matmul(trans_2d_inv,np.array([draw_x,draw_y,np.ones(len(draw_x))]))

        return param,y_loss,draw_x,draw_y

def draw(start_p,mid_p,end_p,zone_start,zone_end,zone_x,zone_y,z,data_x,data_y,data_z,zone_start_i,zone_end_i,save_folder):
    
    # 1. 3d pics
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot3D([start_p[0],zone_start[0]], [start_p[1],zone_start[1]], [start_p[2],zone_start[2]], 'gray')
    ax.plot3D([end_p[0],zone_end[0]], [end_p[1],zone_end[1]], [end_p[2],zone_end[2]], 'gray')
    ax.plot3D(zone_x,zone_y,z,'gray')
    ax.scatter3D(data_x,data_y,data_z,s=1)
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
    ax.plot3D(zone_x,zone_y,z,'gray')
    ax.scatter3D(data_x[zone_start_i:zone_end_i],data_y[zone_start_i:zone_end_i],data_z[zone_start_i:zone_end_i],s=1)
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
    ax.plot(zone_x,zone_y,'gray')
    ax.scatter(data_x[zone_start_i:zone_end_i],data_y[zone_start_i:zone_end_i],s=1)
    ax.scatter(zone_start[0],zone_start[1],c='red')
    ax.scatter(mid_p[0],mid_p[1],c='orange')
    ax.scatter(zone_end[0],zone_end[1],c='green')
    ax.set_xlabel('x-axis (mm)')
    ax.set_ylabel('y-axis (mm)')
    fig.savefig(save_folder+'zone_xy.png')
    # plt.show()
    # 4. zone 2d yz
    plt.close(fig)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(zone_y,z,'gray')
    ax.scatter(data_y[zone_start_i:zone_end_i],data_z[zone_start_i:zone_end_i],s=1)
    ax.scatter(zone_start[1],zone_start[2],c='red')
    ax.scatter(mid_p[1],mid_p[2],c='orange')
    ax.scatter(zone_end[1],zone_end[2],c='green')
    ax.set_xlabel('y-axis (mm)')
    ax.set_ylabel('z-axis (mm)')
    fig.savefig(save_folder+'zone_yz.png')
    # plt.show()
    # 5. zone 2d yz
    plt.close(fig)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(zone_x,z,'gray')
    ax.scatter(data_x[zone_start_i:zone_end_i],data_z[zone_start_i:zone_end_i],s=1)
    ax.scatter(zone_start[0],zone_start[2],c='red')
    ax.scatter(mid_p[0],mid_p[2],c='orange')
    ax.scatter(zone_end[0],zone_end[2],c='green')
    ax.set_xlabel('x-axis (mm)')
    ax.set_ylabel('z-axis (mm)')
    fig.savefig(save_folder+'zone_xz.png')
    # plt.show()
    plt.close(fig)
    plt.clf()

def draw_all_curve(zone_start,mid_p,zone_end,para_x,para_y,cubi_x,cubi_y,quin_x,quin_y,data_x,data_y,save_folder):
    
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(zone_start[0],zone_start[1],c='red')
    ax.scatter(mid_p[0],mid_p[1],c='orange')
    ax.scatter(zone_end[0],zone_end[1],c='green')
    ax.scatter(data_x,data_y,s=1)
    ax.plot(para_x,para_y,'y')
    ax.plot(cubi_x,cubi_y,'c')
    ax.plot(quin_x,quin_y,'m')
    ax.set_xlabel('x-axis (mm)')
    ax.set_ylabel('y-axis (mm)')
    fig.savefig(save_folder+'zone_curve_compare.png')    

def main():
    
    # folder to read
    data_folder = 'data_param/'

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
                move_start = [px-side_l*cos(radians(ang/2)),py+side_l*sin(radians(ang/2)),start_p[2]]
                move_end = [px-side_l*cos(radians(ang/2)),py-side_l*sin(radians(ang/2)),start_p[2]]
                move_mid = [px,py,start_p[2]]

                zone_fit = ZoneFit(joint_angles,move_start,move_mid,move_end,zone)

                print("Line fit")
                d1_3d,d1_2d,d2_3d,d2_2d = zone_fit.L_fit_loss()
                print("Parabola fit")
                param_para,loss_para,draw_x_para,draw_y_para = zone_fit.parabola_fit()
                save_folder=data_folder+'result/'+"log_"+str(vel)+"_"+"{:02d}".format(pxi)+"_"+"{:02d}".format(pyi)+"_"+\
                            str(ang)+"_para_"
                # draw(move_start,move_mid,move_end,zone_fit.zone_start_p,zone_fit.zone_end_p,\
                #     draw_x_para,draw_y_para,np.ones(len(draw_x_para))*move_start[2],\
                #     zone_fit.traj_x,zone_fit.traj_y,zone_fit.traj_z,zone_fit.zone_start_i,zone_fit.zone_end_i,\
                #     save_folder)
                print("Cubic fit")
                param_cubi,loss_cubi,draw_x_cubi,draw_y_cubi = zone_fit.cubic_fit()
                save_folder=data_folder+'result/'+"log_"+str(vel)+"_"+"{:02d}".format(pxi)+"_"+"{:02d}".format(pyi)+"_"+\
                            str(ang)+"_cubi_"
                # draw(move_start,move_mid,move_end,zone_fit.zone_start_p,zone_fit.zone_end_p,\
                #     draw_x_cubi,draw_y_cubi,np.ones(len(draw_x_cubi))*move_start[2],\
                #     zone_fit.traj_x,zone_fit.traj_y,zone_fit.traj_z,zone_fit.zone_start_i,zone_fit.zone_end_i,\
                #     save_folder)
                print("Quintic fit")
                param_quin,loss_quin,draw_x_quin,draw_y_quin = zone_fit.quintic_fit()
                save_folder=data_folder+'result/'+"log_"+str(vel)+"_"+"{:02d}".format(pxi)+"_"+"{:02d}".format(pyi)+"_"+\
                            str(ang)+"_quin_"
                # draw(move_start,move_mid,move_end,zone_fit.zone_start_p,zone_fit.zone_end_p,\
                #     draw_x_quin,draw_y_quin,np.ones(len(draw_x_quin))*move_start[2],\
                #     zone_fit.traj_x,zone_fit.traj_y,zone_fit.traj_z,zone_fit.zone_start_i,zone_fit.zone_end_i,\
                #     save_folder)
                
                # draw compare
                save_folder=data_folder+'result/'+"log_"+str(vel)+"_"+"{:02d}".format(pxi)+"_"+"{:02d}".format(pyi)+"_"+\
                            str(ang)+"_"
                # draw_all_curve(zone_fit.zone_start_p,move_mid,zone_fit.zone_end_p,draw_x_para,draw_y_para,draw_x_cubi,draw_y_cubi,draw_x_quin,draw_y_quin,\
                #     zone_fit.traj_x[zone_fit.zone_start_i:zone_fit.zone_end_i],zone_fit.traj_y[zone_fit.zone_start_i:zone_fit.zone_end_i]\
                #         ,save_folder)
                
                y_loss_zone_cubic[ang][pxi,pyi] = loss_cubi
                y_loss_zone_quintic[ang][pxi,pyi] = loss_quin
                z_height_diff_ave[ang][pxi,pyi] = np.mean(zone_fit.traj_z-start_p[2])
                z_height_diff_std[ang][pxi,pyi] = np.std(zone_fit.traj_z-start_p[2])
                zone_z_height_diff_ave[ang][pxi,pyi] = np.mean(zone_fit.traj_z[zone_fit.zone_start_i:zone_fit.zone_end_i]-start_p[2])
                zone_z_height_diff_std[ang][pxi,pyi] = np.std(zone_fit.traj_z[zone_fit.zone_start_i:zone_fit.zone_end_i]-start_p[2])
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