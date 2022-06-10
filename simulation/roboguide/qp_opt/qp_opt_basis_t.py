from asyncio import current_task
from re import A
from pandas import read_csv, DataFrame
import sys
from copy import deepcopy
sys.path.append('../../../toolbox')
sys.path.append('../../../circular_fit')
from robots_def import *
from utils import *
from toolbox_circular_fit import *
from abb_motion_program_exec_client import *
from robots_def import *
import matplotlib.pyplot as plt
from lambda_calc import *
from qpsolvers import solve_qp
from scipy.optimize import fminbound
from numpy.linalg import norm
from math import sin,cos,pi
from skfda.representation.basis import Fourier,Monomial

def barrier1(x):
    a=10;b=-1;e=0.5;l=5
    return -np.divide(a*b*(x-e),l)

def barrier2(x):
    a=10;b=-1;e=0.1;l=5
    return -np.divide(a*b*(x-e),l)

# robot=abb6640(d=50)
robot=m900ia(d=50)
data_dir='data_qp_devel/'

ang=30
##### create curve #####
###start rotation by 'ang' deg
start_p = np.array([2200,500, 1000])
end_p = np.array([2200, -500, 1000])
mid_p=(start_p+end_p)/2

k=np.cross(end_p+np.array([0.1,0,0])-mid_p,start_p-mid_p)
k=k/np.linalg.norm(k)
theta=np.radians(ang)

R=rot(k,theta)
new_vec=R@(end_p-mid_p)
new_end_p=mid_p+new_vec

###calculate lambda
lam1_f=np.linalg.norm(mid_p-start_p)
lam1=np.linspace(0,lam1_f,num=25001)
lam_f=lam1_f+np.linalg.norm(mid_p-new_end_p)
lam2=np.linspace(lam1_f,lam_f,num=25001)

lam=np.hstack((lam1,lam2[1:]))

try:
    with open(data_dir+'Curve_js.npy','rb') as f:
        curve_js = np.load(f)
    with open(data_dir+'Curve_in_base_frame.npy','rb') as f:
        curve = np.load(f)
    with open(data_dir+'Curve_R_in_base_frame.npy','rb') as f:
        R_all = np.load(f)
        curve_normal=R_all[:,:,-1]
except OSError as e:

    #generate linear segment
    a1,b1,c1=lineFromPoints([lam1[0],start_p[0]],[lam1[-1],mid_p[0]])
    a2,b2,c2=lineFromPoints([lam1[0],start_p[1]],[lam1[-1],mid_p[1]])
    a3,b3,c3=lineFromPoints([lam1[0],start_p[2]],[lam1[-1],mid_p[2]])
    line1=np.vstack(((-a1*lam1-c1)/b1,(-a2*lam1-c2)/b2,(-a3*lam1-c3)/b3)).T

    a1,b1,c1=lineFromPoints([lam2[0],mid_p[0]],[lam2[-1],new_end_p[0]])
    a2,b2,c2=lineFromPoints([lam2[0],mid_p[1]],[lam2[-1],new_end_p[1]])
    a3,b3,c3=lineFromPoints([lam2[0],mid_p[2]],[lam2[-1],new_end_p[2]])
    line2=np.vstack(((-a1*lam2-c1)/b1,(-a2*lam2-c2)/b2,(-a3*lam2-c3)/b3)).T

    curve=np.vstack((line1,line2[1:]))

    R_init=Ry(np.radians(135))
    R_end=Ry(np.radians(90))
    # interpolate orientation
    R_all=[R_init]
    k,theta=R2rot(np.dot(R_end,R_init.T))
    curve_normal=[R_init[:,-1]]
    # theta=np.pi/4 #force 45deg change
    for i in range(1,len(curve)):
        angle=theta*i/(len(curve)-1)
        R_temp=rot(k,angle)
        R_all.append(np.dot(R_temp,R_init))
        curve_normal.append(R_all[-1][:,-1])
    curve_normal=np.array(curve_normal)

    q_init=robot.inv(start_p,R_init)[1]
    #solve inv kin

    curve_js=[q_init]
    for i in range(1,len(curve)):
        q_all=np.array(robot.inv(curve[i],R_all[i]))
        ###choose inv_kin closest to previous joints
        curve_js.append(unwrapped_angle_check(curve_js[-1],q_all))

    curve_js=np.array(curve_js)
    R_all=np.array(R_all)
    with open(data_dir+'Curve_js.npy','wb') as f:
        np.save(f,curve_js)
    with open(data_dir+'Curve_in_base_frame.npy','wb') as f:
        np.save(f,curve)
    with open(data_dir+'Curve_R_in_base_frame.npy','wb') as f:
        np.save(f,R_all)

step = 100
total_seg = int((len(curve_js)-1)/step)
N = total_seg+1

curve = np.hstack((curve,curve_normal))
curve=curve[::step]
curve_js=curve_js[::step]

joint_num = len(curve_js[0])

lam = [0]
for i in range(1,len(curve)):
    lam.append(lam[-1]+norm(curve[i,:3]-curve[i-1,:3]))
lam = np.array(lam)/lam[-1]

ldot_init = 0.01 #1/(1000/10 mm/s) = 0.01
ldot_tar = 0.3
lr = 0.01

# choose basis function
# Fourier basis
bases_num=61

def get_basis_mat(t):
    # the_basis = Fourier((0,1),n_basis=bases_num,period=1)
    the_basis = Monomial(domain_range=(0,t),n_basis=bases_num)
    the_basis_d = the_basis.derivative()
    the_basis_dd = the_basis.derivative(order=2)
    basis_q = the_basis(lam)
    basis_q = np.reshape(basis_q,(bases_num,len(lam))).T
    basis_dq = the_basis_d(lam)
    basis_dq = np.reshape(basis_dq,(bases_num,len(lam))).T
    basis_ddq = the_basis_dd(lam)
    basis_ddq = np.reshape(basis_ddq,(bases_num,len(lam))).T

    basis_mat_col = np.zeros((joint_num*N,bases_num))
    basis_mat_col[::joint_num] = deepcopy(basis_q)
    basis_mat = deepcopy(basis_mat_col)
    basis_dmat_col = np.zeros((joint_num*N,bases_num))
    basis_dmat_col[::joint_num] = deepcopy(basis_dq)
    basis_dmat = deepcopy(basis_dmat_col)
    basis_ddmat_col = np.zeros((joint_num*N,bases_num))
    basis_ddmat_col[::joint_num] = deepcopy(basis_ddq)
    basis_ddmat = deepcopy(basis_ddmat_col)

    alpha_init = None

    for i in range(6):
        alpha_i = np.matmul(np.linalg.pinv(basis_q),curve_js[:,i])

        if i>0:
            basis_mat_col = np.vstack((np.zeros((1,bases_num)),basis_mat_col))
            basis_mat_col = basis_mat_col[:-1]
            basis_mat = np.hstack((basis_mat,basis_mat_col))
            basis_dmat_col = np.vstack((np.zeros((1,bases_num)),basis_dmat_col))
            basis_dmat_col = basis_dmat_col[:-1]
            basis_dmat = np.hstack((basis_dmat,basis_dmat_col))
            basis_ddmat_col = np.vstack((np.zeros((1,bases_num)),basis_ddmat_col))
            basis_ddmat_col = basis_ddmat_col[:-1]
            basis_ddmat = np.hstack((basis_ddmat,basis_ddmat_col))

            alpha_init = np.append(alpha_init,alpha_i)
        else:
            alpha_init = deepcopy(alpha_i)
    
    return basis_mat,basis_dmat,basis_ddmat,alpha_init

qup_path = np.tile(robot.upper_limit,N)
qlow_path = np.tile(robot.lower_limit,N)
qdot_path = np.tile(robot.joint_vel_limit,N)
qddot_path = np.tile(robot.joint_acc_limit,N)

# fig,axs=plt.subplots(1,6)
# for i in range (6):
#     axs[i].scatter(lam,curve_js[:,i],s=1)
#     axs[i].scatter(lam,q_all[:,i],s=1)
#     print(np.max(np.fabs(q_all[:,i]-curve_js[:,i])))
# plt.show()

basis_mat,basis_dmat,basis_ddmat,alpha_init = get_basis_mat(1/ldot_init)

pose_all = np.zeros(6*N)
alpha = deepcopy(alpha_init)
ldot = ldot_init
for i in range(50):
    print("Step:",i)

    basis_mat,basis_dmat,basis_ddmat,_ = get_basis_mat(1/ldot)

    q_all = np.matmul(basis_mat,alpha)
    q_all = np.reshape(q_all,(N,joint_num))

    fig,axs=plt.subplots(1,6)
    for i in range (6):
        axs[i].scatter(lam,curve_js[:,i],s=1)
        axs[i].scatter(lam,q_all[:,i],s=1)
        # print(np.max(np.fabs(q_all[:,i]-curve_js[:,i])))
    plt.show()

    Jp_all=[]
    JR_all=[]
    Jp_mat = np.zeros((3*N,joint_num*N))
    this_lam = [0]
    dldp = []
    last_pose = None
    for n in range(N):
        pose_n = robot.fwd(q_all[n])
        pose_all[6*n:6*n+3] = pose_n.p
        pose_all[6*n+3:6*n+6] = pose_n.R[:,-1]

        J=robot.jacobian(q_all[n])
        Jp_all.append(J[3:])
        Jp_mat[n*3:n*3+3,n*joint_num:n*joint_num+joint_num] = J[3:]
        #modify jacobian here
        JR_all.append(-hat(pose_n.R[:,-1])@J[:3])

        if n>0:
            dp_norm = norm(pose_n.p-last_pose.p)
            this_lam.append(this_lam[-1]+dp_norm)
            dldp = np.append(dldp,dp_norm/(pose_n.p-last_pose.p))
        
        last_pose = pose_n
    # append the last dldp
    dldp = np.append(dldp,dldp[-3:])

    print(this_lam[-1])

    ###path constraint
    diff=curve-pose_all.reshape(N,6)
    distance_p=np.linalg.norm(diff[:,:3],axis=1)
    distance_R=np.linalg.norm(diff[:,3:],axis=1)
    G1=np.zeros((N,joint_num*N))	#position
    G2=np.zeros((N,joint_num*N))	#normal
    for j in range(N):
        G1[j,6*j:6*(j+1)]=(diff[j][:3]/np.linalg.norm(diff[j][:3]))@Jp_all[j]
        G2[j,6*j:6*(j+1)]=(diff[j][3:]/np.linalg.norm(diff[j][3:]))@JR_all[j]

    H = (1/this_lam[-1])*np.matmul(dldp,np.matmul(Jp_mat,basis_dmat))
    P = np.dot(np.reshape(H,(len(H),1)),np.reshape(H,(1,len(H))))+0.1*np.eye(len(H))
    f = -ldot_tar*H

    h1=barrier1(distance_p)
    h2=barrier2(distance_R)
    h=np.hstack((h1,h2))
    G=np.vstack((G1,G2))
    G = np.matmul(G,basis_mat)
    h = h+np.matmul(G,alpha)
    G=-G
    h=-h

    ## joint limit constraint
    G=np.vstack((G,basis_mat))
    h = np.append(h,qup_path)
    # G=np.vstack((G,-basis_mat))
    # h = np.append(h,-qlow_path)
    ## qdot constraint
    G=np.vstack((G,basis_dmat))
    h = np.append(h,qdot_path)
    # ## qddot constraint
    G=np.vstack((G,basis_ddmat))
    h = np.append(h,qddot_path)

    print(np.matmul(H,alpha))

    alpha_tar=solve_qp(P=P,q=f,G=G,h=h)
    d_alpha = alpha_tar-alpha
    alpha = alpha + d_alpha*lr

    ldot = np.matmul(H,alpha)
    print(ldot)
    print(np.matmul(H,alpha_tar))

    break


q_all = np.matmul(basis_mat,alpha)
q_all = np.reshape(q_all,(N,joint_num))
fig,axs=plt.subplots(1,6)
for i in range (6):
    axs[i].scatter(lam,curve_js[:,i],s=1)
    axs[i].scatter(lam,q_all[:,i],s=1)
    # print(np.max(np.fabs(q_all[:,i]-curve_js[:,i])))
plt.show()

curve_exe = []
for n in range(N):
    pose_n = robot.fwd(q_all[n])
    curve_exe.append(pose_n.p)
curve_exe = np.array(curve_exe)

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'red',label='Motion Cmd')
#plot execution curve
ax.plot3D(curve_exe[:,0], curve_exe[:,1],curve_exe[:,2], 'green',label='Executed Motion')
ax.view_init(elev=40, azim=-145)
ax.set_title('Cartesian Interpolation using Motion Cmd')
ax.set_xlabel('x-axis (mm)')
ax.set_ylabel('y-axis (mm)')
ax.set_zlabel('z-axis (mm)')
plt.show()