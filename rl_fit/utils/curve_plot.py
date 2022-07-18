import sys
from pandas import *
sys.path.append('../../toolbox')
from robots_def import *
from general_robotics_toolbox import *
from error_check import *

robot=abb6640(d=50)

# curve_idx=100
# step_size=500
# curve_file = "train_data/base/traj_{}.csv".format(curve_idx)
# curve_js_file = "train_data/js_new/traj_{}_js_new.csv".format(curve_idx)

# curve = read_csv(curve_file).values[::step_size]
# curve_js = read_csv(curve_js_file).values[::step_size]


# curve_fwd=[]
# for i in range(len(curve)):
# 	curve_fwd.append(robot.fwd(curve_js[i]).p)
# curve_fwd=np.array(curve_fwd)

# plt.figure()
# plt.plot(np.linalg.norm(curve_fwd-curve[:,:3],axis=1))

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='red')
# ax.plot3D(curve_fwd[:,0], curve_fwd[:,1], curve_fwd[:,2], c='blue')
# plt.show()


idx=1
base_file = "train_data/poly/base/curve_base_poly_{}.csv".format(idx)
js_file = "train_data/poly/js/curve_js_poly_{}.csv".format(idx)
lam_file = "train_data/poly/lambda/lambda_{}.csv".format(idx)
###read in poly
col_names=['poly_x', 'poly_y', 'poly_z','poly_direction_x', 'poly_direction_y', 'poly_direction_z']
data = read_csv(base_file, names=col_names)
poly_x=data['poly_x'].tolist()
poly_y=data['poly_y'].tolist()
poly_z=data['poly_z'].tolist()
curve_poly_coeff=np.vstack((poly_x, poly_y, poly_z))

col_names=['poly_q1', 'poly_q2', 'poly_q3','poly_q4', 'poly_q5', 'poly_q6']
data = read_csv(js_file, names=col_names)
poly_q1=data['poly_q1'].tolist()
poly_q2=data['poly_q2'].tolist()
poly_q3=data['poly_q3'].tolist()
poly_q4=data['poly_q4'].tolist()
poly_q5=data['poly_q5'].tolist()
poly_q6=data['poly_q6'].tolist()
curve_js_poly_coeff=np.vstack((poly_q1, poly_q2, poly_q3,poly_q4,poly_q5,poly_q6))

lam = read_csv(lam_file, header=None).values.flatten()

curve_poly=[]
curve_js_poly=[]
for i in range(3):
	curve_poly.append(np.poly1d(curve_poly_coeff[i]))
for i in range(len(robot.joint_vel_limit)):
	curve_js_poly.append(np.poly1d(curve_js_poly_coeff[i]))

###get curve based on lambda
curve=np.vstack((curve_poly[0](lam),curve_poly[1](lam),curve_poly[2](lam))).T
curve_js=np.vstack((curve_js_poly[0](lam),curve_js_poly[1](lam),curve_js_poly[2](lam),curve_js_poly[3](lam),curve_js_poly[4](lam),curve_js_poly[5](lam))).T

curve_fwd=[]
for i in range(len(curve)):
	curve_fwd.append(robot.fwd(curve_js[i]).p)
curve_fwd=np.array(curve_fwd)

plt.figure()
plt.plot(np.linalg.norm(curve_fwd-curve[:,:3],axis=1))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='red')
ax.plot3D(curve_fwd[:,0], curve_fwd[:,1], curve_fwd[:,2], c='blue')
plt.show()