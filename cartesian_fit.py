import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import *
from pwlfmd import *



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
	# df=DataFrame({'x':np.flip(xHat),'y':np.flip(pred[:,0]), 'z':np.flip(pred[:,1])})
	# df.to_csv('fit.csv')

if __name__ == "__main__":
	main()