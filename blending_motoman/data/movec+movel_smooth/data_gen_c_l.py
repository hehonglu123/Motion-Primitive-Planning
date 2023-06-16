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
from exe_toolbox import *
from lambda_calc import *


# dataset='movec+movel_smooth'

robot=abb6640(d=50)
###generate a continuous arc, with linear orientation
start_p = np.array([2200,500, 1000])
mid_p = np.array([2500,0,1000])
end_p = np.array([2200, -500, 1000])

c,r=circle_from_3point(start_p,end_p,mid_p)
arc=arc_from_3point(start_p,end_p,mid_p,N=5001)[:2501]



lam=calc_lam_cs(arc)
avg_distance=np.average(np.gradient(lam))
N_plane=np.cross(c-start_p,c-arc[-1])
l_vec=np.cross(c-arc[-1],N_plane)
l_vec=l_vec/np.linalg.norm(l_vec)
line=arc[-1]+np.outer(lam[1:],l_vec)

curve=np.vstack((arc,line))

#get orientation only
q_init=np.array([0.3120923 ,  0.27688859,  0.40240991,  0.6676647 , -0.49953974,
       -0.52210179])
R_init=robot.fwd(q_init).R
R_end=Ry(np.radians(90))
#interpolate orientation and solve inv kin
curve_js=[q_init]
R_all=[R_init]
k,theta=R2rot(np.dot(R_end,R_init.T))
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

# ###########save to csv####################
df=DataFrame({'breakpoints':np.array([0,int((len(curve)+1)/2),len(curve)]),'primitives':['movej_fit','movec_fit','movel_fit'],'points':[[q_init],[curve[int(len(curve)/4)],curve[int((len(curve)+1)/2)]],[curve[-1]]]})
df.to_csv('command.csv',header=True,index=False)


df=DataFrame({'x':curve[:,0],'y':curve[:,1], 'z':curve[:,2],'x_dir':R_all[:,0,-1],'y_dir':R_all[:,1,-1], 'z_dir':R_all[:,2,-1]})
df.to_csv('Curve_in_base_frame.csv',header=False,index=False)
DataFrame(curve_js).to_csv('Curve_js.csv',header=False,index=False)