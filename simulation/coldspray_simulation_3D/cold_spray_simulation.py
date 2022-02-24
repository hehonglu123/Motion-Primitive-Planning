from math import cos, erf, exp, pi, sqrt
import numpy as np
from numpy import deg2rad
import csv
from stl import mesh

import sys
# from toolbox.robot_def import fwd
sys.path.append('../../toolbox')
from robots_def import *

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

class PointBaseMesh(object):
    def __init__(self,mesh) -> None:
        super().__init__()

        self.points = np.array([])
        self.normals = []
        self.mesh_num = []
        self.ver_num = []
        for mi in range(len(mesh.data)):
            this_normal = mesh.data[mi][0]
            this_vertex = mesh.data[mi][1]
            
            for vi in range(len(this_vertex)):
                this_point = this_vertex[vi]
                exist = False
                
                for pti in range(len(self.points)):
                    if np.linalg.norm(self.points[pti]-this_point) < 1e-10:
                        exist=True
                        break
                
                if exist:
                    self.normals[pti] = np.vstack((self.normals[pti],this_normal))
                    self.mesh_num[pti] = np.append(self.mesh_num[pti],mi)
                    self.ver_num[pti] = np.append(self.ver_num[pti],vi)
                else:
                    if len(self.points) != 0:
                        self.points = np.vstack((self.points,this_point))
                    else:
                        self.points = np.array(this_point)
                    self.normals.append(this_normal)
                    self.mesh_num.append(np.array([mi]))
                    self.ver_num.append(np.array([vi]))
                

    def update_mesh(self,mesh):
        
        for pti in range(len(self.points)):

            for mvi in range(len(self.mesh_num[pti])):
                mesh_num = self.mesh_num[pti][mvi]
                ver_num = self.ver_num[pti][mvi]
                mesh.data[mesh_num][1][ver_num] = self.points[pti]
        
        mesh.update_normals()
        return mesh

class Spray_Simulator(object):
    def __init__(self,mold_file,a=18,sigma=14,check_ang_margin=88) -> None:
        super().__init__()

        self.mesh = mesh.Mesh.from_file(mold_file)
        self.point_mesh = PointBaseMesh(self.mesh)

        # parameters
        self.a = a
        self.sigma = sigma
        self.check_ang_margin = check_ang_margin
    
    def layer_spray(self,j_stamp,joint_p,robot):

        for n in range(len(joint_p)-1):
            this_pose = robot.fwd(np.deg2rad(joint_p[n]))
            next_pose = robot.fwd(np.deg2rad(joint_p[n+1]))

            # compute the nozzle orientation and duration
            nz = -this_pose.R[:,2]
            nx = next_pose.p-this_pose.p-np.dot((next_pose.p-this_pose.p),nz)*nz
            nx = nx/np.linalg.norm(nx)
            ny = np.cross(nz,nx)
            ny = ny/np.linalg.norm(ny)
            ln = np.dot(nx,(next_pose.p-this_pose.p))
            delta_t = j_stamp[n+1]-j_stamp[n]

            # point impact angle check
            im_ang = self.impact_angle(nz)

            # loop all the points
            for pti in range(len(self.point_mesh.points)):
                if im_ang[pti] == 0:
                    continue
                point = self.point_mesh.points[pti]
                R = np.array([nx,ny,nz])
                pn_noz = np.matmul(R,(point-this_pose.p))
                g1 = self.a/(2*sqrt(2*pi)*self.sigma*ln)
                g2 = exp(-(pn_noz[1]^2)/(2*self.sigma^2))
                g3 = erf(pn_noz[0]/(sqrt(2)*self.sigma))-erf((pn_noz[0]-ln)/(sqrt(2)*self.sigma))
                gp = g1*g2*g3*nz
                self.point_mesh.points[pti] = point+gp*delta_t
        
        self.mesh = self.point_mesh.update_mesh(self.mesh)
        return self.mesh

    def impact_angle(self, nz):

        im_ang = np.array([])
        for pti in range(len(self.point_mesh.points)):
            im_ang = np.append(im_ang,0)
            for nl in self.point_mesh.normals[pti]:
                if np.dot(nl,nz) > cos(deg2rad(self.check_ang_margin)):
                    im_ang[-1] = 1
                    break
        return im_ang


def main():
    
    # your_mesh = mesh.Mesh.from_file('test_stl_compare.stl')
    # print("Normals: ",your_mesh.normals)
    # print("V0",your_mesh.v0)
    # print("V1",your_mesh.v1)
    # print("V2",your_mesh.v2)
    
    sim = Spray_Simulator('test_stl.stl')
    # sim = Spray_Simulator('test_stl_compare.stl')

    path_file = 'log_cold_spray_lines.csv'

    # read Robotstudio logged data
    with open(path_file,"r") as f:
        rows = csv.reader(f, delimiter=',')

        log_results_dict = {}
        for col in rows:
            if len(log_results_dict) == 0:
                log_results_dict['timestamp']=[]
                log_results_dict['cmd_num']=[]
                log_results_dict['joint_angle']=[]
                continue
            log_results_dict['timestamp'].append(float(col[0]))
            log_results_dict['cmd_num'].append(int(col[1]))
            log_results_dict['joint_angle'].append(np.deg2rad(np.array([float(col[2]),float(col[3]),float(col[4]),float(col[5]),float(col[6]),float(col[7])])))
        stamps = log_results_dict['timestamp']
        cmd_num = log_results_dict['cmd_num']
        joint_angles = log_results_dict['joint_angle']
    
    robot = abb6640()

    start_cmd = 3
    start_ji = 0
    for cmdnum in cmd_num:
        if start_cmd == cmdnum:
            break
        start_ji += 1
    stamps = stamps[start_ji:]
    joint_angles = joint_angles[start_ji:]

    new_mesh = sim.layer_spray(stamps,joint_angles,robot)

    # Create a new plot
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)

    # Load the STL files and add the vectors to the plot
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(new_mesh.vectors))

    # Auto scale to the mesh size
    scale = new_mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    plt.show()

if __name__=='__main__':
    main()