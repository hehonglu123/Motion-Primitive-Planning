import numpy as np
import time, sys
import matplotlib.pyplot as plt
from pandas import *
from general_robotics_toolbox import *
sys.path.append('../../../toolbox')
from utils import *
from robots_def import *

dataset='movel_smooth'
robot=robot_obj('MA2010_A0',def_path='../../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../../config/weldgun2.csv',\
    pulse2deg_file_path='../../../config/MA2010_A0_pulse2deg.csv',d=50)

start_p = np.array([1300,500, 300])
end_p = np.array([1300, -500, 300])
mid_p=(start_p+end_p)/2

lam_f=np.linalg.norm(end_p-start_p)
lam=np.linspace(0,lam_f,num=5001)


#generate linear segment
a1,b1,c1=lineFromPoints([lam[0],start_p[0]],[lam[-1],end_p[0]])
a2,b2,c2=lineFromPoints([lam[0],start_p[1]],[lam[-1],end_p[1]])
a3,b3,c3=lineFromPoints([lam[0],start_p[2]],[lam[-1],end_p[2]])
curve=np.vstack(((-a1*lam-c1)/b1,(-a2*lam-c2)/b2,(-a3*lam-c3)/b3)).T

#get orientation
R=np.array([[ -1, 0, -0.    ],
	[0, 1,  0.    ],
	[0.,      0.,     -1.    ]])
q_init=robot.inv(start_p,R,np.zeros(6))[0]
R_init=robot.fwd(q_init).R
R_end=np.array([[ -1, 0, 0    ],
				[0, 0,  1.    ],
				[0,1., 0.    ]])
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

visualize_curve_w_normal(curve,R_all[:,:,-1],100)
# ###########save to csv####################
df=DataFrame({'breakpoints':np.array([0,int((len(curve)+1)/2),len(curve)]),'primitives':['moveabsj_fit','movel_fit','movel_fit'],'p_bp':[[curve[0]],[curve[int((len(curve)+1)/2)]],[curve[-1]]],'q_bp':[[curve_js[0]],[curve_js[int((len(curve_js)+1)/2)]],[curve_js[-1]]]})
df.to_csv('command.csv',header=True,index=False)

df=DataFrame({'x':curve[:,0],'y':curve[:,1], 'z':curve[:,2],'x_dir':R_all[:,0,-1],'y_dir':R_all[:,1,-1], 'z_dir':R_all[:,2,-1]})
df.to_csv('Curve_in_base_frame.csv',header=False,index=False)
DataFrame(curve_js).to_csv('Curve_js.csv',header=False,index=False)