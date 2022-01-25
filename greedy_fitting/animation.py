import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pandas import *
import sys
sys.path.append('../circular_fit')
from toolbox_circular_fit import *

i=-1
breakpoints=np.array([[2314.15978241,  958.42826816,  702.08826496],
						[2250.3019386 ,  712.27412173,  674.6589419],
						[2183.26402417,  521.15792021,  695.23669442],
						[2100.66907832,  272.14741655,  737.38589596],
						[2072.09256109,  155.71712879,  749.82359435],
						[2047.97738171,  -91.45488056,  735.10741075],
						[2093.83211381, -300.05601747,  639.76162908],
						[2179.79676751, -435.45833991,  513.00900606],
						[2215.37701947, -474.87408667,  463.00787292]])

col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
data = read_csv("../data/from_ge/Curve_backproj_in_base_frame.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve=np.vstack((curve_x, curve_y, curve_z)).T

col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
data = read_csv("comparison/moveL+moveC/threshold1/curve_fit_backproj.csv")
curve_x=data['x'].tolist()
curve_y=data['y'].tolist()
curve_z=data['z'].tolist()
curve_fit=np.vstack((curve_x, curve_y, curve_z)).T

curve=curve[::100]
curve_fit=curve_fit[::100]

breakpoints=np.array([83,152,200,251,318,390,451,479,500,-1])

curve_fit=curve_fit[:breakpoints[i]]

###plane projection visualization
curve_mean = curve.mean(axis=0)
curve_centered = curve - curve_mean
U,s,V = np.linalg.svd(curve_centered)
# Normal vector of fitting plane is given by 3rd column in V
# Note linalg.svd returns V^T, so we need to select 3rd row from V^T
normal = V[2,:]

curve_2d_vis = rodrigues_rot(curve_centered+curve_mean, normal, [0,0,1])[:,:2]
curve_fit_2d_vis = rodrigues_rot(curve_fit, normal, [0,0,1])[:,:2]


x = curve_fit_2d_vis[:,0]
y = curve_fit_2d_vis[:,1]

fig, ax = plt.subplots()
line, = ax.plot(x, y, color='k')

def update(num, x, y, line):
    line.set_data(x[:num], y[:num])
    line.axes.axis([-1300, -500, 200, 1800])
    return line,

ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, y, line],
                              interval=25, blit=True)
ani.save('images/test'+str(i)+'.gif')
# plt.show()