import numpy as np
from stl import mesh
from pandas import *
from copy import deepcopy
import yaml

data_dir='from_NX/'
stl_dir='../simulation/gazebo_sim/model/blade/meshes/'
blade_pose_dir='../simulation/roboguide/data/curve_blade/'
output_dir='blade_shift/'

############### relative path
relative_path=read_csv(data_dir+"Curve_dense.csv",header=None).values
relative_path=np.array(relative_path)
print(relative_path[0,:3])
relative_path_new = deepcopy(relative_path)
relative_path_new[:,:3] = relative_path_new[:,:3]-relative_path[0,:3]

# print(relative_path[:5])
# print(relative_path_new[:5])

DataFrame(relative_path_new).to_csv(output_dir+'Curve_dense.csv',header=False,index=False)

############### blade pose from baseline
with open(blade_pose_dir+'blade_pose.yaml') as file:
    blade_pose = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
blade_pose_new = deepcopy(blade_pose)
# print(blade_pose_new)
blade_pose_new[:3,3]=blade_pose[:3,3]+np.matmul(blade_pose[:3,:3],relative_path[0,:3])
# print(blade_pose_new)
with open(output_dir+'blade_pose_baseline.yaml','w') as file:
    yaml.dump({'H':blade_pose_new.tolist()},file)

############### Mesh
# Using an existing stl file:
mesh_origin = mesh.Mesh.from_file(stl_dir+'generic_fan_blade.stl')

# print(mesh_origin.v0[:3])
# print(mesh_origin.v1[:3])
# print(mesh_origin.v2[:3])

mesh_new = deepcopy(mesh_origin)
mesh_new.v0 = mesh_new.v0-relative_path[0,:3]
mesh_new.v1 = mesh_new.v1-relative_path[0,:3]
mesh_new.v2 = mesh_new.v2-relative_path[0,:3]
mesh_new.save(output_dir+'blade_shift.stl')

# print(relative_path[0,:3])
# print(mesh_origin.v0[:3])
# print(mesh_new.v0[:3])
# print(mesh_new.vectors[:3])