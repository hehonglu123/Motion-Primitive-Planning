import numpy as np
from matplotlib import pyplot as plt

lam=np.load("D:/Motion-Primitive-Planning/simulation/roboguide/data/baseline_m10ia/curve_blade_scale/greedy_20/lambda.npy")
error=np.load("D:/Motion-Primitive-Planning/simulation/roboguide/data/baseline_m10ia/curve_blade_scale/greedy_20/error.npy")
angle_error=np.load("D:/Motion-Primitive-Planning/simulation/roboguide/data/baseline_m10ia/curve_blade_scale/greedy_20/normal_error.npy")
speed=np.load("D:/Motion-Primitive-Planning/simulation/roboguide/data/baseline_m10ia/curve_blade_scale/greedy_20/speed.npy")

fig, ax1 = plt.subplots(figsize=(6,4))
ax2 = ax1.twinx()
ax1.plot(lam, speed, 'g-', label='Speed')
ax2.plot(lam, error, 'b-',label='Error')
ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
draw_speed_max=max(speed)*1.05
ax1.axis(ymin=0,ymax=draw_speed_max)
draw_error_max=max(np.append(error,np.degrees(angle_error)))*1.05
ax2.axis(ymin=0,ymax=draw_error_max)

ax1.set_xlabel('lambda (mm)')
ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
plt.title("Speed and Error Plot")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc=1)

plt.show()