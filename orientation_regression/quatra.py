import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from pandas import *
import sys
sys.path.append('../toolbox')
from robots_def import *
from direction2R import *
from general_robotics_toolbox import *
from error_check import *



col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
data = read_csv("../data/from_ge/curve_backproj_js.csv", names=col_names)
curve_q1=data['q1'].tolist()
curve_q2=data['q2'].tolist()
curve_q3=data['q3'].tolist()
curve_q4=data['q4'].tolist()
curve_q5=data['q5'].tolist()
curve_q6=data['q6'].tolist()
curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

robot=abb6640()

curve_quat=[]
curve_R=[]
for i in range(len(curve_js)):
	curve_R.append(robot.fwd(curve_js[i]).R)
	curve_quat.append(R2q(curve_R[-1]))

start_idx=30000
Q=np.array(curve_quat)[start_idx:start_idx+10000].T
Z=np.dot(Q,Q.T)
u, s, vh = np.linalg.svd(Z)

w=np.dot(quatproduct(u[:,1]),quatcomplement(u[:,0]))
k,theta=q2rot(w)

theta1=2*np.arctan2(np.dot(u[:,1],curve_quat[start_idx]),np.dot(u[:,0],curve_quat[start_idx]))
theta2=2*np.arctan2(np.dot(u[:,1],curve_quat[start_idx+9999]),np.dot(u[:,0],curve_quat[start_idx+9999]))

theta=(theta2-theta1)%(2*np.pi)
if theta>np.pi:
	theta-=2*np.pi
print(theta1,theta2)
print(k,theta)

R=np.dot(curve_R[start_idx+9999],curve_R[start_idx].T)
print(R2rot(R))
