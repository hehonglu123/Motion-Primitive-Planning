import numpy as np
import sys, time, cv2
sys.path.append('../toolbox')
from projection import LinePlaneCollision
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

######################################All in mm###################################
###coldspray 3d simulation
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
	###length:x, width: y
	def __init__(self,width,length,resolution,p,R):
		###parameters of surface to be deposit to
		self.width=width
		self.length=length
		self.resolution=resolution
		self.p=p
		self.R=R
		self.deposition_image=np.zeros((round(width/resolution),round(length/resolution)))
	def clear_deposition(self):
		self.deposition_image=np.zeros((round(self.width/resolution),round(self.length/resolution)))
		return

class environment(object):
	def __init__(self,nozzle,surface,d):
		###initialize tool and surface
		self.surface=surface
		self.nozzle=nozzle
		self.d=d
		###initialize simulation graphics
		# plt.ion()
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111, projection='3d')
		cv2.namedWindow("Image")
	def update_graphics(self):
		# x_data=np.tile(np.arange(0,self.surface.width,self.surface.resolution),round(self.surface.length/self.surface.resolution))
		# y_data=np.tile(np.arange(0,self.surface.length,self.surface.resolution),(round(self.surface.width/self.surface.resolution),1)).T.flatten()
		# self.ax.bar3d(x_data,y_data,np.zeros(len(self.surface.deposition_image.flatten())),self.surface.resolution, self.surface.resolution, self.surface.deposition_image.flatten())
		
		normalizedImg = np.zeros((round(self.surface.width/self.surface.resolution),round(self.surface.length/self.surface.resolution)))
		normalizedImg = cv2.normalize(self.surface.deposition_image,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
		cv2.imshow('Image', self.surface.deposition_image)
		cv2.waitKey(1)

	def generate_spray_path(self,path_on_surface):
		###generate nozzle trajectory given path on surface and distance d
		return path_on_surface+self.d*self.surface.R[:,-1]

	def single_shoot(self):
		###single shot deposition onto the surface
		intersection=LinePlaneCollision(planeNormal=self.surface.R[:,-1], planePoint=self.surface.p, rayDirection=self.nozzle.R[:,-1], rayPoint=self.nozzle.p)
		coord_on_surface=np.dot(self.surface.R.T,intersection-self.surface.p)
		###centroid shift from center to top left corner
		bins_coord=coord_on_surface+np.array([self.surface.length/2.,self.surface.width/2.,0])
		bins_coord=np.round(bins_coord/self.surface.resolution).astype(int)

		###generate gaussian distribution deposition
		Y, X = np.ogrid[:len(self.surface.deposition_image), :len(self.surface.deposition_image[0])]
		dist_from_center = np.sqrt((X - bins_coord[0])**2 + (Y-bins_coord[1])**2)
		mask = dist_from_center <= self.nozzle.radius
		self.surface.deposition_image+=mask*self.nozzle.feed_rate

	def spray(self,traj_p,traj_R):
		###spray along a given nozzle path
		for i in range(len(traj_p)):
			self.nozzle.p=traj_p[i]
			self.nozzle.R=traj_R[i]
			self.single_shoot()
			self.update_graphics()

def main():
	noz=nozzle(1,0.5,1)
	width=100
	length=100
	resolution=0.1
	surf=surface(width,length,resolution,np.zeros(3),np.eye(3))
	env=environment(noz,surf,5)

	path_on_surface=np.zeros((int(length/resolution),3))
	path_on_surface[:,0]=np.arange(-width/2,width/2,resolution)
	traj_p=env.generate_spray_path(path_on_surface)
	traj_R=[-np.eye(3)]*len(path_on_surface)

	env.spray(traj_p,traj_R)
	print('physics finished')


	
	cv2.destroyAllWindows()

	# env.update_graphics()
	# plt.show()

	# print(surf.deposition_image)
	# animation.FuncAnimation(env.fig, env.update_graphics, 25,
 #                                   interval=50, blit=False)
	# plt.show()



if __name__ == "__main__":
	main()
			