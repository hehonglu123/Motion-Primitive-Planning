import numpy as np
from stl import mesh
from pandas import *
from copy import deepcopy
import yaml

data_dir='from_NX/'
stl_dir='../simulation/gazebo_sim/model/blade/meshes/'
output_dir='from_NX_scale/'

############### relative path
relative_path=read_csv(data_dir+"Curve_dense.csv",header=None).values
relative_path=np.array(relative_path)
print(relative_path[0,:3])
relative_path_new = deepcopy(relative_path)

scale_ratio = 0.8
############### scale curve
relative_path_new[:,:3] = scale_ratio*relative_path_new[:,:3]
print(relative_path_new[0,:3])

############### save curve
DataFrame(relative_path_new).to_csv(output_dir+'Curve_dense.csv',header=False,index=False)

############### Mesh
# Using an existing stl file:
mesh_origin = mesh.Mesh.from_file(stl_dir+'generic_fan_blade.stl')

mesh_new = deepcopy(mesh_origin)
mesh_new.v0 = scale_ratio*mesh_new.v0
mesh_new.v1 = scale_ratio*mesh_new.v1
mesh_new.v2 = scale_ratio*mesh_new.v2
mesh_new.save(output_dir+'blade_scale.stl')