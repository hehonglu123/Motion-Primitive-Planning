import matplotlib.pyplot as plt
import numpy as np
from pandas import *
import sys
sys.path.append('../toolbox')

from lambda_calc import *

def main():
	dataset='wood/'
	curve = read_csv("../train_data/"+dataset+"/Curve_in_base_frame.csv",header=None).values[:,:3]
	curve_js = read_csv("../train_data/"+dataset+"/Curve_js.csv",header=None).values[:]
	curve_fit_js = read_csv("greedy_output/curve_fit_js.csv",header=None).values
	# curve_fit_js = read_csv("../train_data/"+dataset+"greedy_output/0.5/curve_fit_js.csv",header=None).values

	lam=calc_lam_cs(curve)

	dqdlam_fit=np.divide(np.gradient(curve_fit_js,axis=0),np.tile(np.gradient(lam),(6,1)).T)
	ddqdlam_fit=np.gradient(dqdlam_fit,axis=0)

	plt.figure()
	for i in range(len(curve_js[0])):
		plt.subplot(2,3,i+1)
		plt.plot(lam,curve_js[:,i],label='original')
		plt.plot(lam,curve_fit_js[:,i],label='fit')
		plt.title('J'+str(i+1))
	plt.legend()
	plt.figure()

	for i in range(len(curve_js[0])):
		plt.subplot(2,3,i+1)
		plt.plot(lam,ddqdlam_fit[:,i])
		plt.title('J'+str(i+1))
	plt.suptitle('δ δq/δlam',fontsize=20)
	plt.figure()

	for i in range(len(curve_js[0])):
		plt.subplot(2,3,i+1)
		plt.plot(lam,dqdlam_fit[:,i])
		plt.title('J'+str(i+1))
	plt.suptitle('δq/δlam',fontsize=20)
	plt.show()
if __name__ == "__main__":
	main()