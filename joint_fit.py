import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import *
from pwlfmd import *


def main():
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("data/Curve_js.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	###x ref, yz out
	my_pwlf=MDFit(curve[:,0],curve[:,1:])

	###arbitrary breakpoints
	# break_points=[np.min(curve[:,0]),500,1000,np.max(curve[:,0])]

	###slope calc breakpoints
	# break_points=my_pwlf.x_data[my_pwlf.break_slope()]
	# my_pwlf.fit_with_breaks(break_points)

	###fit by error thresholding
	my_pwlf.fit_under_error_simplified(0.01)

	###predict for the determined points
	q1Hat = np.linspace(np.min(curve[:,0]),np.max(curve[:,0]), num=1000)
	pred = my_pwlf.predict(q1Hat)

	print('maximum error: ',my_pwlf.calc_max_error())

if __name__ == "__main__":
	main()