import sys, time
sys.path.append('../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *
import matplotlib.animation as animation

def main():
	global ys,ax1
	

	dataset='wood/'

	data_dir="../data/"+dataset
	fitting_output="../data/"+dataset+'baseline/100L/'


	curve_js=read_csv(data_dir+'Curve_js.csv',header=None).values
	curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values

	fig=plt.figure()
	plt.plot(curve[:,0])
	fig.canvas.manager.window.move(0,0)
	plt.show(block=False)
	plt.pause(0.01)
	for i in range(10):
		time.sleep(0.1)
	plt.close(fig)

	fig=plt.figure()
	plt.plot(curve[:,0])
	plt.plot(curve[:,1])
	fig.canvas.manager.window.move(800,0)
	plt.show(block=False)
	plt.pause(0.1)
	for i in range(10):
		time.sleep(0.1)
	plt.close(fig)



if __name__ == '__main__':
	main()