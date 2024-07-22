import numpy as np


def Cartesian2Spherical(n):
	# Convert Cartesian coordinates to spherical coordinates
	
	theta = np.arctan2(np.sqrt(n[0]**2 + n[1]**2), n[2])
	phi = np.arctan2(n[1], n[0])
	
	return np.array([theta, phi])

n=[-0.,-0.,-1.]
print(Cartesian2Spherical(n))