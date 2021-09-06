import numpy as np
import traceback, copy, sys
from scipy.optimize import differential_evolution
from scipy.optimize import fmin_l_bfgs_b
from scipy import linalg
sys.path.append('toolbox')
from error_check import *
from robot_def import *


###multi dimension piece-wise linear fit
class MDFit(object):
	###input: reference 1-D [] double lam, other dimension N-D [] double data   
	def __init__(self, lam, data):
		self.lam_data=lam
		self.data=data
		self.lapack_driver='gelsd'
		self.break_points=[0,-1]

	def clear_breakpoints(self):
		self.break_points=[0,-1]

	def assemble_regression_matrix(self, breaks, lam):

		
		# Sort the breaks, then store them
		breaks_order = np.argsort(breaks)
		self.fit_breaks = breaks[breaks_order]
		# store the number of parameters and line segments
		self.n_segments = len(breaks) - 1

		# Assemble the regression matrix
		A_list = [np.ones_like(lam)]

		A_list.append(lam - self.fit_breaks[0])
		for i in range(self.n_segments - 1):
			A_list.append(np.where(lam > self.fit_breaks[i+1],
								   lam - self.fit_breaks[i+1],
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
			data_range[-1]=len(self.lam_data)
		for i in range(data_range[0]+step_size,data_range[-1],step_size):
			###calc slope vector of both groups
			slope=self.calc_slope(np.hstack((np.reshape(self.lam_data[break_point_idx:i+1],(-1,1)),self.data[break_point_idx:i+1])).T)
			next_slope=self.calc_slope(np.hstack((np.reshape(self.lam_data[i:i+step_size+1],(-1,1)),self.data[i:i+step_size+1])).T)
			###trigger breakpoint by dotproduct of 2 eig
			if 1-abs(np.dot(slope,next_slope))>min_threshold:
				###if sharp turn
				if 1-abs(np.dot(slope,next_slope))>max_threshold and step_size>3:
					###smooth out by adding more breakpoints
					temp_break_slope=self.break_slope(data_range=(break_point_idlam,i+step_size),step_size=int(step_size/2))
				else:
					break_points.append(i)
					break_point_idx=i
		if break_points[-1]+1!=len(self.lam_data):

			break_points.append(-1)

		return break_points

	def break_slope_simplified(self,min_threshold=0.05):
		for i in range(1,len(self.lam_data)-1):

			###calc slope vector of both groups
			vec1=np.hstack((self.lam_data[i],self.data[i]))-np.hstack((self.lam_data[i-1],self.data[i-1]))
			slope = vec1 / np.linalg.norm(vec1)

			vec2=np.hstack((self.lam_data[i+1],self.data[i+1]))-np.hstack((self.lam_data[i],self.data[i]))
			next_slope = vec2 / np.linalg.norm(vec2)
			###trigger breakpoint by dotproduct of 2 slope vector

			if 1-np.dot(slope,next_slope)>min_threshold:	#valued from 0 to 2
				self.break_points.append(i)


		self.break_points.sort()
		self.break_points.append(self.break_points.pop(0))

	def lstsq(self, A):
		"""
		Perform the least squares fit for A matrix.
		"""
		beta, ssr, _, _ = linalg.lstsq(A, self.data,
									   lapack_driver=self.lapack_driver)

		return beta



	def fit_with_breaks(self, breaks):
		print('fitting')
		self.A = self.assemble_regression_matrix(np.array(breaks), self.lam_data)

		# try to solve the regression problem
		try:
			print('solving lstsq')
			self.beta = self.lstsq(self.A)

		except:
			traceback.print_exc()
		print('fitting finished')
		return

	def predict(self):
		# solve the regression problem
		data_hat = np.dot(self.A, self.beta)
		return data_hat

	def predict_arb(self, lam):

		lam = np.array(lam)

		A = self.assemble_regression_matrix(self.lam_data[self.break_points], lam)

		# solve the regression problem
		data_hat = np.dot(A, self.beta)
		return data_hat

	def calc_max_error1(self):
		#calculate worst case error at each index in 3D space
		print('calculating error')

		fit=self.predict()

		errors=fit-self.data
		
		max_error=np.max(np.linalg.norm(errors,axis=1))

		print('error calculating finished')
		return max_error

	def calc_max_error2(self):
		#calculate worst case error at each index in 3D space from joint space
		print('calculating error')

		fit=self.predict()
		if len(self.data[0])>3:
			#if in joint space
			fit_cartesian=[]
			curve_cartesian=[]
			for i in range(len(fit)):
				fit_cartesian.append(fwd(fit[i]).p)
				curve_cartesian.append(fwd(self.data[i]).p)
			return calc_max_error(fit_cartesian,curve_cartesian)
		else:
			return calc_max_error(fit,self.data)

	def fit_under_error(self,max_error):
		min_threshold=0.3
		step_size=10
		break_points=self.lam_data[self.break_slope(min_threshold=min_threshold,step_size=step_size)]
		self.fit_with_breaks(break_points)
		error=self.calc_max_error1()
		while error>max_error:
			###tune slope threshold first, then step size
			if min_threshold<0.01:
				if step_size==1:
					###terminate at finest fitting
					print('finest fitting')
					min_threshold=-1
					break_points=self.lam_data[self.break_slope(min_threshold=min_threshold,step_size=step_size)]
					self.fit_with_breaks(break_points)
					error=self.calc_max_error1()
					return error
				else:
					step_size=int(step_size/2)
			else:
				min_threshold=min_threshold/2

			break_points=self.lam_data[self.break_slope(min_threshold=min_threshold,step_size=step_size)]
			self.fit_with_breaks(break_points)
			error=self.calc_max_error1()
		return error

	def fit_under_error_simplified(self,max_error=0.1,starting_threshold=0.05):
		min_threshold=starting_threshold
		self.break_slope_simplified(min_threshold)
		self.fit_with_breaks(self.lam_data[self.break_points])
		error=self.calc_max_error1()
		print('error: ',error,'num breakpoints: ',len(self.break_points), 'threshold: ',min_threshold)
		while error>max_error:
			###tune slope threshold first, then step size
			if min_threshold<0.0000001:
		
				print('finest fitting')
				min_threshold=-1
				self.break_slope_simplified(min_threshold)
				self.fit_with_breaks(self.lam_data[self.break_points])
				error=self.calc_max_error1()
				print('error: ',error,'num breakpoints: ',len(self.break_points), 'threshold: ',min_threshold)
				return error
			else:
				min_threshold=min_threshold/2

			self.break_slope_simplified(min_threshold)
			print(self.break_points)
			self.fit_with_breaks(self.lam_data[self.break_points])
			error=self.calc_max_error1()
			print('error: ',error,'num breakpoints: ',len(self.break_points), 'threshold: ',min_threshold)
		return error
