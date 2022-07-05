import sys, time
sys.path.append('../../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *
import matplotlib.animation as animation

dataset='wood/'

data_dir="../../data/"+dataset
fitting_output="../../data/"+dataset+'baseline/100L/'


curve_js=read_csv(data_dir+'Curve_js.csv',header=None).values
curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values



fig = plt.figure()
#creating a subplot 
ax1 = fig.add_subplot(1,1,1)
start=time.time()
def animate(i):
	xs=range(0,len(curve))

	v=np.random.rand()

	ax1.clear()
	ax1.plot(xs, curve[:,0],label="traj1")
	if time.time()-start>5:
		ax1.plot(xs, curve[:,1],label="traj2")

	plt.legend()
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Live graph with matplotlib')	
	
	
ani = animation.FuncAnimation(fig, animate, interval=1000) 
plt.show()