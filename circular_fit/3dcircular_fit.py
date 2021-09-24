import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from pandas import *
from scipy.optimize import minimize

def generate_circle_by_vectors(t, C, r, n, u):
    n = n/np.linalg.norm(n)
    u = u/np.linalg.norm(u)
    P_circle = r*np.cos(t)[:,np.newaxis]*u + r*np.sin(t)[:,np.newaxis]*np.cross(n,u) + C
    return P_circle

def fit_circle_2d_constraint(data):
    data=data.T
    fun = lambda x: np.linalg.norm(x[0]*data[0] + x[1]*data[1] + data[0][0]**2 - x[0]*data[0][0] + data[1][0]**2 - x[1]*data[1][0] - np.square(data[0]) - np.square(data[1]))
    res = minimize(fun, (0,0), method='SLSQP')

    center=res.x/2.
    r=np.sqrt((data[0][0]-center[0])**2+(data[1][0]-center[1])**2)
    return center[0], center[1], r

def fit_circle_2d(x, y, w=[]):
    
    A = np.array([x, y, np.ones(len(x))]).T
    b = x**2 + y**2
    
    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W,A)
        b = np.dot(W,b)
    
    # Solve by method of least squares
    c = np.linalg.lstsq(A,b,rcond=None)[0]
    
    # Get circle parameters from solution c
    xc = c[0]/2
    yc = c[1]/2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r


#-------------------------------------------------------------------------------
# RODRIGUES ROTATION
# - Rotate given points based on a starting and ending vector
# - Axis k and angle of rotation theta given by vectors n0,n1
#   curve_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))
#-------------------------------------------------------------------------------
def rodrigues_rot(curve, n0, n1):
    
    # If curve is only 1d array (coords of single point), fix it to be matrix
    if curve.ndim == 1:
        curve = curve[np.newaxis,:]
    
    # Get vector of rotation k and angle theta
    n0 = n0/np.linalg.norm(n0)
    n1 = n1/np.linalg.norm(n1)
    k = np.cross(n0,n1)
    k = k/np.linalg.norm(k)
    theta = np.arccos(np.dot(n0,n1))
    
    # Compute rotated points
    curve_rot = np.zeros((len(curve),3))
    for i in range(len(curve)):
        curve_rot[i] = curve[i]*np.cos(theta) + np.cross(k,curve[i])*np.sin(theta) + k*np.dot(k,curve[i])*(1-np.cos(theta))

    return curve_rot


#-------------------------------------------------------------------------------
# ANGLE BETWEEN
# - Get angle between vectors u,v with sign based on plane with unit normal n
#-------------------------------------------------------------------------------
def angle_between(u, v, n=None):
    if n is None:
        return np.arctan2(np.linalg.norm(np.cross(u,v)), np.dot(u,v))
    else:
        return np.arctan2(np.dot(n,np.cross(u,v)), np.dot(u,v))

    
#-------------------------------------------------------------------------------
# - Make axes of 3D plot to have equal scales
# - This is a workaround to Matplotlib's set_aspect('equal') and axis('equal')
#   which were not working for 3D
#-------------------------------------------------------------------------------
def set_axes_equal_3d(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = np.abs(limits[:,0] - limits[:,1])
    centers = np.mean(limits, axis=1)
    radius = 0.5 * np.max(spans)
    ax.set_xlim3d([centers[0]-radius, centers[0]+radius])
    ax.set_ylim3d([centers[1]-radius, centers[1]+radius])
    ax.set_zlim3d([centers[2]-radius, centers[2]+radius])


###read in points
col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
data = read_csv("../data/from_interp/Curve_in_base_frame.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve=np.vstack((curve_x, curve_y, curve_z)).T

break_point=int(len(curve)/2)
curve1=curve[:break_point]
curve2=curve[break_point:]

curve=curve

fig = figure(figsize=(15,11))
alpha_pts = 0.5
figshape = (2,3)
ax = [None]*4
ax[0] = subplot2grid(figshape, loc=(0,0), colspan=2)
ax[1] = subplot2grid(figshape, loc=(1,0))
ax[2] = subplot2grid(figshape, loc=(1,1))
ax[3] = subplot2grid(figshape, loc=(1,2))
i = 0
ax[i].set_title('Fitting circle in 2D coords projected onto fitting plane')
ax[i].set_xlabel('x'); ax[i].set_ylabel('y');
ax[i].set_aspect('equal', 'datalim'); ax[i].margins(.1, .1); ax[i].grid()
i = 1
ax[i].plot(curve[:,0], curve[:,1], 'y-', lw=3, label='Generating circle')
ax[i].scatter(curve[:,0], curve[:,1], alpha=alpha_pts, label='Cluster points P')
ax[i].set_title('View X-Y')
ax[i].set_xlabel('x'); ax[i].set_ylabel('y');
ax[i].set_aspect('equal', 'datalim'); ax[i].margins(.1, .1); ax[i].grid()
i = 2
ax[i].plot(curve[:,0], curve[:,2], 'y-', lw=3, label='Generating circle')
ax[i].scatter(curve[:,0], curve[:,2], alpha=alpha_pts, label='Cluster points P')
ax[i].set_title('View X-Z')
ax[i].set_xlabel('x'); ax[i].set_ylabel('z'); 
ax[i].set_aspect('equal', 'datalim'); ax[i].margins(.1, .1); ax[i].grid()
i = 3
ax[i].plot(curve[:,1], curve[:,2], 'y-', lw=3, label='Generating circle')
ax[i].scatter(curve[:,1], curve[:,2], alpha=alpha_pts, label='Cluster points P')
ax[i].set_title('View Y-Z')
ax[i].set_xlabel('y'); ax[i].set_ylabel('z'); 
ax[i].set_aspect('equal', 'datalim'); ax[i].margins(.1, .1); ax[i].grid()
#-------------------------------------------------------------------------------
# (1) Fitting plane by SVD for the mean-centered data
# Eq. of plane is <p,n> + d = 0, where curve is a point on plane and n is normal vector
#-------------------------------------------------------------------------------
curve_mean = curve.mean(axis=0)
curve_centered = curve - curve_mean
U,s,V = np.linalg.svd(curve_centered)

# Normal vector of fitting plane is given by 3rd column in V
# Note linalg.svd returns V^T, so we need to select 3rd row from V^T
normal = V[2,:]
d = -np.dot(curve_mean, normal)  # d = -<p,n>

#-------------------------------------------------------------------------------
# (2) Project points to coords X-Y in 2D plane
#-------------------------------------------------------------------------------
curve_xy = rodrigues_rot(curve_centered, normal, [0,0,1])

ax[0].scatter(curve_xy[:,0], curve_xy[:,1], alpha=alpha_pts, label='Projected points')

#-------------------------------------------------------------------------------
# (3) Fit circle in new 2D coords
#-------------------------------------------------------------------------------
xc, yc, r = fit_circle_2d_constraint(curve_xy)

#--- Generate circle points in 2D
t = np.linspace(0, 2*np.pi, 100)
xx = xc + r*np.cos(t)
yy = yc + r*np.sin(t)

ax[0].plot(xx, yy, 'k--', lw=2, label='Fitting circle')
ax[0].plot(xc, yc, 'k+', ms=10)
ax[0].legend()

#-------------------------------------------------------------------------------
# (4) Transform circle center back to 3D coords
#-------------------------------------------------------------------------------
C = rodrigues_rot(np.array([xc,yc,0]), [0,0,1], normal) + curve_mean
C = C.flatten()

#--- Generate points for fitting circle
t = np.linspace(0, 2*np.pi, 100)
u = curve[0] - C
curve_fitcircle = generate_circle_by_vectors(t, C, r, normal, u)

ax[1].plot(curve_fitcircle[:,0], curve_fitcircle[:,1], 'k--', lw=2, label='Fitting circle')
ax[2].plot(curve_fitcircle[:,0], curve_fitcircle[:,2], 'k--', lw=2, label='Fitting circle')
ax[3].plot(curve_fitcircle[:,1], curve_fitcircle[:,2], 'k--', lw=2, label='Fitting circle')
ax[3].legend()

#--- Generate points for fitting arc
u = curve[0] - C
v = curve[-1] - C
theta = angle_between(u, v, normal)

t = np.linspace(0, theta, 100)
curve_fitarc = generate_circle_by_vectors(t, C, r, normal, u)

ax[1].plot(curve_fitarc[:,0], curve_fitarc[:,1], 'k-', lw=3, label='Fitting arc')
ax[2].plot(curve_fitarc[:,0], curve_fitarc[:,2], 'k-', lw=3, label='Fitting arc')
ax[3].plot(curve_fitarc[:,1], curve_fitarc[:,2], 'k-', lw=3, label='Fitting arc')
ax[1].plot(C[0], C[1], 'k+', ms=10)
ax[2].plot(C[0], C[2], 'k+', ms=10)
ax[3].plot(C[1], C[2], 'k+', ms=10)
ax[3].legend()

plt.show()
print('Fitting plane: n = %s' % np.array_str(normal, precision=4))
print('Fitting circle: center = %s, r = %.4g' % (np.array_str(C, precision=4), r))
print('Fitting arc: u = %s, θ = %.4g' % (np.array_str(u, precision=4), theta*180/np.pi))