import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
from scipy.optimize import fminbound
sys.path.append('../toolbox')
from lambda_calc import *
from utils import *
# from data.baseline import pose_opt, curve_frame_conversion, find_js

R = 25.4 * 2
H = 25.4 * 1
W = 30
#38x89mm

###generate curve for 1.5x3.5 parabola
def find_point(t, a=0.54, b=1000., c=0.5):
    fr = a * np.pi / 180.0 * (t/b) ** c

    x = t
    y = W * np.sin(np.multiply(fr,x))
    z = (R ** 2 - y ** 2) * H / R ** 2

    return np.vstack((x,y,z)).T

def find_normal(p):
    nx = np.zeros((1,len(p[:,0])))

    ny=np.ones(len(p))
    nz=-2*H*p[:,1]/R**2
    nz=-1/nz
    ###nan protection
    ny[np.isinf(nz)] = 0
    nz[np.isinf(nz)] = 1
    ###normalize
    curve_normal=np.vstack((nx,ny,nz)).T
    curve_normal=np.divide(curve_normal,np.tile(np.linalg.norm(curve_normal,axis=1),(3,1)).T)
    idx=np.where(curve_normal[:,-1]>0)
    curve_normal[idx]=-curve_normal[idx]
    return curve_normal

def distance_calc(t,p,step_size):
    p_next=find_point(t)
    return np.abs(step_size-np.linalg.norm(p-p_next))

def find_next_point(t,p,step_size):
    t_next=fminbound(distance_calc,t,t+step_size,args=(p,step_size))
    p_next=find_point(t_next)
    normal_next=find_normal(p_next)
    return t_next, p_next, normal_next


def generate_rl_data(robot, data_size=200, reverse=False, show=False):
    save_dir = 'train_data/curve1/reverse' if reverse else 'train_data/curve1/forward'
    # initial_point = np.array([1090.1612137174207, -506.72253554211704, 994.0466002214232, -0.14380380418973931, -0.00396817476110772, -0.9895982616646132])
    for curve_idx in range(data_size):
        print("{:>5} / {:>5}".format(curve_idx, data_size))
        a = min(max(np.random.normal(0.5, 1.), 0.2), 0.7)
        b = min(max(np.random.normal(1000, 5.), 990), 1010)
        c = min(max(np.random.normal(2., 1), 1.5), 2.5)

        t = np.linspace(0, 1000, 1000)
        curve = find_point(t, a, b, c)
        curve_normal = find_normal(curve)

        xsurf = np.linspace(0, 1000, 100)
        ysurf = np.linspace(- R, R, 100)
        zsurf = np.zeros((len(xsurf), len(ysurf)))
        for i in range(len(ysurf)):
            zsurf[:, i] = (R ** 2 - ysurf[i] ** 2) * H / R ** 2

        if show:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.plot_surface(xsurf, ysurf, zsurf, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
            ax.plot3D(curve[:, 0], curve[:, 1], curve[:, 2], 'r.-')
            ax.quiver(curve[::20, 0], curve[::20, 1], curve[::20, 2], curve_normal[::20, 0], curve_normal[::20, 1],
                      curve_normal[::20, 2], length=5, normalize=True)
            plt.title('Curve 1')
            plt.show()

        num_points = 5000
        lam = calc_lam_cs(curve)
        lam = np.linspace(0, lam[-1], num_points)
        curve_act = [curve[0]]
        curve_normal_act = [curve_normal[0]]
        t_act = [0]
        lam_act = np.linspace(0, lam[-1], num_points)
        for i in range(1, num_points):
            t_next, p_next, normal_next = find_next_point(t_act[-1], curve_act[-1], lam[i] - lam[i - 1])
            curve_act.append(p_next.flatten())
            curve_normal_act.append(normal_next.flatten())
            t_act.append(t_next)

        curve_act = np.array(curve_act)
        curve_normal_act = np.array(curve_normal_act)

        if show:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.plot3D(curve_act[:, 0], curve_act[:, 1], curve_act[:, 2], 'r.-')
            plt.show()

        # curve_act = np.flip(curve_act, 0) if reverse else curve_act
        # curve_normal_act = np.flip(curve_normal_act, 0) if reverse else curve_normal_act
        #
        # H_pose = pose_opt(robot, curve_act, curve_normal_act)
        # curve_base, curve_normal_base = curve_frame_conversion(curve_act, curve_normal_act, H_pose)
        #
        # curve_js_all = find_js(robot, curve_base, curve_normal_base)
        # J_min = []
        # for i in range(len(curve_js_all)):
        #     J_min.append(find_j_min(robot, curve_js_all[i]))
        #
        # J_min = np.array(J_min)
        # curve_js = curve_js_all[np.argmin(J_min.min(axis=1))]
        #
        # if show:
        #     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        #     ax.plot3D(curve_base[:, 0], curve_base[:, 1], curve_base[:, 2], 'r.-')
        #     plt.show()
        #
        # df_base = DataFrame({'x':curve_base[:,0],'y':curve_base[:,1], 'z':curve_base[:,2],'x_dir':curve_normal_base[:,0],'y_dir':curve_normal_base[:,1], 'z_dir':curve_normal_base[:,2]})
        # df_base.to_csv(save_dir + os.sep + 'base/curve_{}.csv'.format(curve_idx), header=False, index=False)
        #
        # df_js = DataFrame(curve_js)
        # df_js.to_csv(save_dir + os.sep + 'js/curve_{}.csv'.format(curve_idx), header=False, index=False)


def main():

    t = np.linspace(0,1000,1000)
    curve=find_point(t)
    curve_normal=find_normal(curve)

    ##############3D plots####################
    xsurf = np.linspace(0,1000,100)
    ysurf = np.linspace(- R,R,100)
    zsurf = np.zeros((len(xsurf),len(ysurf)))
    for i in range(len(ysurf)):
        zsurf[:,i] = (R ** 2 - ysurf[i] ** 2) * H / R ** 2

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(xsurf, ysurf, zsurf, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.plot3D(curve[:,0],curve[:,1],curve[:,2],'r.-')
    ax.quiver(curve[::20,0],curve[::20,1],curve[::20,2],curve_normal[::20,0],curve_normal[::20,1],curve_normal[::20,2],length=5, normalize=True)
    plt.title('Curve 1')
    plt.show()

    ###################check curve normal##############################
    # curve_tan=np.gradient(curve,axis=0)
    # print(curve_tan)
    # for i in range(len(curve_tan)):
    #     print(curve_tan[i]@curve_normal[i])
    # plt.plot(np.diag(np.inner(curve_tan,curve_normal)))
    # plt.show()
    # lam=calc_lam_cs(curve)
    # diff=np.linalg.norm(np.diff(curve,axis=0),axis=1)
    # plt.plot(diff)
    # plt.show()

    ####################generate equally spaced points##########################
    num_points=5000
    lam=calc_lam_cs(curve)
    lam=np.linspace(0,lam[-1],num_points)
    curve_act=[curve[0]]
    curve_normal_act=[curve_normal[0]]
    t_act=[0]
    lam_act=np.linspace(0,lam[-1],num_points)
    for i in range(1,num_points):
        t_next, p_next, normal_next=find_next_point(t_act[-1],curve_act[-1],lam[i]-lam[i-1])
        curve_act.append(p_next.flatten())
        curve_normal_act.append(normal_next.flatten())
        t_act.append(t_next)

    curve_act=np.array(curve_act)
    curve_normal_act=np.array(curve_normal_act)

    # DataFrame(np.hstack((curve_act,curve_normal_act))).to_csv('Curve_dense.csv',header=False,index=False)

    # diff=np.linalg.norm(np.diff(curve_act,axis=0),axis=1)
    # plt.figure()
    # plt.plot(diff)
    # plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot3D(curve_act[:,0],curve_act[:,1],curve_act[:,2],'r.-')
    ax.quiver(curve_act[:,0],curve_act[:,1],curve_act[:,2],curve_normal_act[:,0],curve_normal_act[:,1],curve_normal_act[:,2],length=1, normalize=True)
    plt.show()

    visualize_curve_w_normal(curve_act,curve_normal_act,stepsize=10)


if __name__ == '__main__':
    # main()
    robot = abb6640(d=50)
    generate_rl_data(robot, data_size=5, reverse=False, show=True)
