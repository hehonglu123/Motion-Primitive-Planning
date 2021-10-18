import numpy as np
sys.path.append('../toolbox')
from projection import LinePlaneCollision
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

class nozzle(object):
	def __init__(self,radius,gauss_std,feed_rate):
		###cold spray tool initialization
		self.radius=radius
		self.gauss_std=gauss_std
		self.feed_rate=feed_rate
		self.p=np.zeros(3)
		self.R=np.eye(3)
	def update_pose(self,p,R):
		self.p=p
		self.R=R
		return


class surface(object):
	def __init__(self,width,length,resolution,p,R):
		###parameters of surface to be deposit to
		self.width=width
		self.length=length
		self.resolution=resolution
		self.p=p
		self.R=R
		self.deposition_image=np.zeros((round(length/resolution),round(width/resolution)))
	def clear_deposition(self):
		self.deposition_image=np.zeros((round(length/resolution),round(width/resolution)))
		return

class env(object):
	def __init__(self,nozzle,surface,d):
		###initialize tool and surface
		self.surface=surface
		self.nozzle=nozzle
		###initialize simulation graphics
		self.fig = plt.figure()
		self.ax = p3.Axes3D(fig)
	def generate_spray_path(self,path_on_surface,d):
		###generate nozzle trajectory given path on surface and distance d
		return path_on_surface-d*self.surface.R[:,-1]

	def single_shoot(self):
		###single shot deposition onto the surface
		intersection=LinePlaneCollision(planeNormal=self.surface.R[:,-1], planePoint=self.surface.p, rayDirection=self.nozzle.R[:,-1], rayPoint=self.nozzle.p)
		coord_on_surface=np.dot(self.surface.R.T,intersection-self.surface.p)
		bins_coord=np.round(coord_on_surface/resolution).astype(int)
		###generate gaussian distribution deposition
		Y, X = np.ogrid[:len(self.surface.deposition_image), :len(self.surface.deposition_image[0])]
		dist_from_center = np.sqrt((X - bins_coord[0])**2 + (Y-bins_coord[1])**2)
    	mask = dist_from_center <= radius
    	self.surface.deposition_image+=mask*1

	def spray(self,traj_p,traj_R):
		###spray along a given nozzle path
		for i in range(len(traj_p)):
			self.single_shoot()
			