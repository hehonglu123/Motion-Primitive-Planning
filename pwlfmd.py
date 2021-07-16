import numpy as np
from pandas import *
import traceback, copy
from scipy.optimize import differential_evolution
from scipy.optimize import fmin_l_bfgs_b
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


###multi dimension piece-wise linear fit
class MDFit(object):
	###input: reference 1-D [] double x, other dimension N-D [] double data   
	def __init__(self, x, data):
		self.x_data=x
		self.data=data
		self.lapack_driver='gelsd'

	def assemble_regression_matrix(self, breaks, x):

		
		# Sort the breaks, then store them
		breaks_order = np.argsort(breaks)
		self.fit_breaks = breaks[breaks_order]
		# store the number of parameters and line segments
		self.n_segments = len(breaks) - 1

		# Assemble the regression matrix
		A_list = [np.ones_like(x)]

		A_list.append(x - self.fit_breaks[0])
		for i in range(self.n_segments - 1):
			A_list.append(np.where(x > self.fit_breaks[i+1],
								   x - self.fit_breaks[i+1],
									   0.0))

		A = np.vstack(A_list).T

		return A

	### calculate slope vector
	def calc_slope(self,data):
		if len(data[0])<=1:
			###if only 1 point
			return np.ones(len(data))
		cov=np.cov(data)
		w,v=np.linalg.eig(cov)
		eigv=v[np.argmax(w)]
		return eigv
	def break_slope(self,data_range=[0,-1],min_threshold=0.05,max_threshold=0.3,step_size=3):
		
		break_points=[0]
		break_point_idx=0

		if data_range[-1]==-1:
			data_range[-1]=len(self.x_data)
		for i in range(data_range[0]+step_size,data_range[-1],step_size):
			###calc slope vector of both groups
			slope=self.calc_slope(np.hstack((np.reshape(self.x_data[break_point_idx:i+1],(-1,1)),self.data[break_point_idx:i+1])).T)
			next_slope=self.calc_slope(np.hstack((np.reshape(self.x_data[i:i+step_size+1],(-1,1)),self.data[i:i+step_size+1])).T)
			###trigger breakpoint by dotproduct of 2 eig
			if 1-abs(np.dot(slope,next_slope))>min_threshold:
				###if sharp turn
				if 1-abs(np.dot(slope,next_slope))>max_threshold and step_size>3:
					###smooth out by adding more breakpoints
					temp_break_slope=self.break_slope(data_range=(break_point_idx,i+step_size),step_size=int(step_size/2))
				else:
					break_points.append(i)
					break_point_idx=i
		if break_points[-1]+1!=len(self.x_data):

			break_points.append(-1)

		return break_points


	def lstsq(self, A):
		"""
		Perform the least squares fit for A matrix.
		"""
		beta, ssr, _, _ = linalg.lstsq(A, self.data,
									   lapack_driver=self.lapack_driver)

		return beta



	def fit_with_breaks(self, breaks):

		A = self.assemble_regression_matrix(np.array(breaks), self.x_data)

		# try to solve the regression problem
		try:
			self.beta = self.lstsq(A)

		except:
			traceback.print_exc()

		return

	def predict(self, x):

		x = np.array(x)

		A = self.assemble_regression_matrix(self.fit_breaks, x)

		# solve the regression problem
		data_hat = np.dot(A, self.beta)
		return data_hat

	def calc_max_error(self):
		max_error=0
		for i in range(len(self.x_data)):
			pred=self.predict(self.x_data[i])
			error=np.linalg.norm(pred-self.data[i])
			if error>max_error:
				max_error=copy.deepcopy(error)
		return max_error

	def fit_under_error(self,max_error):
		min_threshold=0.3
		step_size=10
		break_points=self.x_data[self.break_slope(min_threshold=min_threshold,step_size=step_size)]
		self.fit_with_breaks(break_points)
		error=self.calc_max_error()
		while error>max_error:
			###tune slope threshold first, then step size
			if min_threshold<0.01:
				if step_size==1:
					###terminate at finest fitting
					print('finest fitting')
					min_threshold=-1
					break_points=self.x_data[self.break_slope(min_threshold=min_threshold,step_size=step_size)]
					self.fit_with_breaks(break_points)
					error=self.calc_max_error()
					return error
				else:
					step_size=int(step_size/2)
			else:
				min_threshold=min_threshold/2

			break_points=self.x_data[self.break_slope(min_threshold=min_threshold,step_size=step_size)]
			self.fit_with_breaks(break_points)
			error=self.calc_max_error()
		return error


def main():
	col_names=['X', 'Y', 'Z','r', 'p', 'y'] 
	data = read_csv("../Curve.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T

	###x ref, yz out
	my_pwlf=MDFit(curve[:,0],curve[:,1:])

	###arbitrary breakpoints
	# break_points=[np.min(curve[:,0]),500,1000,np.max(curve[:,0])]

	###slope calc breakpoints
	# break_points=my_pwlf.x_data[my_pwlf.break_slope()]
	# my_pwlf.fit_with_breaks(break_points)

	###fit by error thresholding
	my_pwlf.fit_under_error(1)

	###predict for the determined points
	xHat = np.linspace(np.min(curve[:,0]),np.max(curve[:,0]), num=1000)
	pred = my_pwlf.predict(xHat)

	print('maximum error: ',my_pwlf.calc_max_error())

	#plot results
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot3D(xHat, pred[:,0], pred[:,1], 'gray')
	ax.scatter3D(curve_x, curve_y, curve_z, c=curve_z, cmap='Accent');
	
	plt.show()

	###output to csv
	df=DataFrame({'x':np.flip(xHat),'y':np.flip(pred[:,0]), 'z':np.flip(pred[:,1])})
	df.to_csv('fit.csv')

if __name__ == "__main__":
	main()