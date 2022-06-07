import numpy as np
from pandas import *
import sys, traceback, time
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
sys.path.append('../../toolbox')
from robots_def import *

robot=abb6640()
curve_js = read_csv('trajectory/arm2.csv',header=None).values
curve=[]
for i in range(len(curve_js)):
	curve.append(robot.fwd(curve_js[i]).p)

curve=np.array(curve)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(curve[:,0], curve[:,1], curve[:,2],label='original',c='gray')
ax.legend()
plt.show()