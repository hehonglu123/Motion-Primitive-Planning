import numpy as np

class nozzle(object):
	def __init__(self,diameter,gauss_std,feed_rate):
		###cold spray tool initialization
		self.diameter=diameter
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
	def __init__(self,nozzle,surface):
		###initialize tool and surface
		self.surface=surface
		self.nozzle=nozzle
		###initialize simulation graphics

	def generate_spray_path(self,path_on_surface,d):
		###generate nozzle trajectory given path on surface and distance d

	def single_shoot(self):
		###single shot deposition onto the surface

	def spray(self,traj_p,traj_R):
		###spray along a given nozzle path
		for i in range(len(traj_p)):
			self.single_shoot()