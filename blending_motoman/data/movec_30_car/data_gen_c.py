import numpy as np
import time, sys
import matplotlib.pyplot as plt
from pandas import *
from general_robotics_toolbox import *
sys.path.append('../')
sys.path.append('../../toolbox')
from toolbox_circular_fit import *
from utils import *
from robots_def import *
from lambda_calc import *

robot=abb6640(d=50)
###generate a continuous arc, with linear orientation
start_p = np.array([2200,500, 1000])
mid_p = np.array([2500,0,1000])
end_p = np.array([2200, -500, 1000])

#get original arc
arc=arc_from_3point(start_p,end_p,mid_p,N=5001)
arc1=arc[:int((len(arc)+1)/2)]
arc2=arc[int((len(arc)+1)/2):]
###start rotation by 30 deg
k=np.cross(end_p-arc2[0],start_p-arc2[0])
k=k/np.linalg.norm(k)
theta=np.radians(30)
R=rot(k,theta)

arc2_new=np.zeros(arc2.shape)
for i in range(len(arc2)):
	new_vec=R@(arc2[i]-arc2[0])
	arc2_new[i]=arc2[0]+new_vec

curve=np.vstack((arc1,arc2_new))

#get orientation
q_init=np.array([0.3120923 ,  0.27688859,  0.40240991,  0.6676647 , -0.49953974,
       -0.52210179])
R_init=robot.fwd(q_init).R
R_end=Ry(np.radians(90))
#interpolate orientation and solve inv kin
curve_js=[q_init]
R_all=[R_init]
k,theta=R2rot(np.dot(R_end,R_init.T))
# theta=np.pi/4 #force 45deg change
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
print('total length: ',calc_lam_cs(curve)[-1])
visualize_curve_w_normal(curve,R_all[:,:,-1],50)
###########save to csv####################
df=DataFrame({'breakpoints':np.array([0,int((len(curve)+1)/2),len(curve)]),'primitives':['movej_fit','movec_fit','movec_fit'],'points':[[q_init],[curve[int(len(curve)/4)],curve[int((len(curve)+1)/2)]],[curve[int(3*len(curve)/4)],curve[-1]]]})
df.to_csv('command.csv',header=True,index=False)


df=DataFrame({'x':curve[:,0],'y':curve[:,1], 'z':curve[:,2],'x_dir':R_all[:,0,-1],'y_dir':R_all[:,1,-1], 'z_dir':R_all[:,2,-1]})
df.to_csv('Curve_in_base_frame.csv',header=False,index=False)
DataFrame(curve_js).to_csv('Curve_js.csv',header=False,index=False)