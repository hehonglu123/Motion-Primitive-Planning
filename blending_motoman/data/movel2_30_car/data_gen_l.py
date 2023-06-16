import numpy as np
import time, sys
import matplotlib.pyplot as plt
from pandas import *
from general_robotics_toolbox import *
sys.path.append('../../toolbox')
from utils import *
from robots_def import *

robot=abb6640(d=50)

start_p = np.array([2376.26152,	1089.256029,	746.5836202])
mid_p = np.array([2296.10018593,  692.24076104,  669.3971284])
end_p	=start_p+(mid_p-start_p)*2

###start rotation by 30 deg
k=np.cross(end_p+np.array([0.1,0,0])-mid_p,start_p-mid_p)
k=k/np.linalg.norm(k)
theta=np.radians(30)

R=rot(k,theta)
new_vec=R@(end_p-mid_p)
end_p=mid_p+new_vec

# print(end_p)

###calculate lambda
lam1_f=np.linalg.norm(mid_p-start_p)
lam1=np.linspace(0,lam1_f,num=2501)
lam_f=lam1_f+np.linalg.norm(mid_p-end_p)
lam2=np.linspace(lam1_f,lam_f,num=2501)

lam=np.hstack((lam1,lam2[1:]))


#generate linear segment
a1,b1,c1=lineFromPoints([lam1[0],start_p[0]],[lam1[-1],mid_p[0]])
a2,b2,c2=lineFromPoints([lam1[0],start_p[1]],[lam1[-1],mid_p[1]])
a3,b3,c3=lineFromPoints([lam1[0],start_p[2]],[lam1[-1],mid_p[2]])
line1=np.vstack(((-a1*lam1-c1)/b1,(-a2*lam1-c2)/b2,(-a3*lam1-c3)/b3)).T

a1,b1,c1=lineFromPoints([lam2[0],mid_p[0]],[lam2[-1],end_p[0]])
a2,b2,c2=lineFromPoints([lam2[0],mid_p[1]],[lam2[-1],end_p[1]])
a3,b3,c3=lineFromPoints([lam2[0],mid_p[2]],[lam2[-1],end_p[2]])
line2=np.vstack(((-a1*lam2-c1)/b1,(-a2*lam2-c2)/b2,(-a3*lam2-c3)/b3)).T

curve=np.vstack((line1,line2[1:]))


#get orientation
q_init=np.array([0.626837286,	0.839988113,	-0.245742828,	1.700793354,	-0.899330476,	0.768529957])
q_end=np.array([0.575135423,	0.921567943,	-0.147742612,	1.569344818,	-1.375486144,	0.605301197])
R_init=robot.fwd(q_init).R
R_end=robot.fwd(q_end).R
#interpolate orientation and solve inv kin
curve_js=[q_init]
R_all=[R_init]
k,theta=R2rot(np.dot(R_end,R_init.T))
# theta=np.pi/2 #force 90deg change

for i in range(1,len(curve)):
	angle=theta*i/(len(curve)-1)
	R_temp=rot(k,angle)
	R_all.append(np.dot(R_temp,R_init))
	q_all=np.array(robot.inv(curve[i],R_all[-1]))
	###choose inv_kin closest to previous joints
	temp_q=q_all-curve_js[-1]
	order=np.argsort(np.linalg.norm(temp_q,axis=1))
	curve_js.append(q_all[order[0]])

curve_js=np.array(curve_js)
R_all=np.array(R_all)

visualize_curve_w_normal(curve,R_all[:,:,-1],50)
# ###########save to csv####################
df=DataFrame({'breakpoints':np.array([0,int((len(curve)+1)/2),len(curve)]),'primitives':['movej_fit','movel_fit','movel_fit'],'points':[[q_init],[curve[int((len(curve)+1)/2)]],[curve[-1]]]})
df.to_csv('command.csv',header=True,index=False)

df=DataFrame({'x':curve[:,0],'y':curve[:,1], 'z':curve[:,2],'x_dir':R_all[:,0,-1],'y_dir':R_all[:,1,-1], 'z_dir':R_all[:,2,-1]})
df.to_csv('Curve_in_base_frame.csv',header=False,index=False)
DataFrame(curve_js).to_csv('Curve_js.csv',header=False,index=False)