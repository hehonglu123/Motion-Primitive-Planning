import numpy as np

def generate_circle_by_vectors(t, C, r, n, u):
    n = n/np.linalg.norm(n)
    u = u/np.linalg.norm(u)
    P_circle = r*np.cos(t)[:,np.newaxis]*u + r*np.sin(t)[:,np.newaxis]*np.cross(n,u) + C
    return P_circle

def fit_circle_2d(x, y, p=[]):
    if len(p)==0:
        A = np.array([x, y, np.ones(len(x))]).T
        b = x**2 + y**2
        
        # Solve by method of least squares
        c = np.linalg.lstsq(A,b,rcond=None)[0]
        
        # Get circle parameters from solution c
        xc = c[0]/2
        yc = c[1]/2
        r = np.sqrt(c[2] + xc**2 + yc**2)
        return xc, yc, r
    else:

        ###rewrite lstsq to fit point p on circle
        A = np.array([x-p[0], y-p[1]]).T
        b = x**2 + y**2 - p[0]**2 - p[1]**2
        
        # Solve by method of least squares
        c = np.linalg.lstsq(A,b,rcond=None)[0]
        
        # Get circle parameters from solution c
        xc = c[0]/2
        yc = c[1]/2
        r=np.linalg.norm(p[:-1]-np.array([xc,yc]))
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

def circle_fit(curve,p=[]):
    ###curve: 3D point data
    ###p:   constraint point of the arc
    ########################################
    ###return curve_fit: 3D point data

    if len(p)!=0:
        ###fit on a plane first
        curve_mean = curve.mean(axis=0)
        curve_centered = curve - curve_mean
        p_centered = p - curve_mean

        ###constraint fitting
        ###rewrite lstsq to fit point p on plane
        A = np.array([curve_centered[:,0]-p_centered[0]*curve_centered[:,2]/p_centered[2], curve_centered[:,1]-p_centered[1]*curve_centered[:,2]/p_centered[2]]).T
        b = np.ones(len(curve))-curve_centered[:,2]/p_centered[2]
        c = np.linalg.lstsq(A,b,rcond=None)[0]
        normal=np.array([c[0],c[1],(1-c[0]*p_centered[0]-c[1]*p_centered[1])/p_centered[2]])

        

        ###make sure constraint point is on plane
        # print(np.dot(normal,p_centered))
        ###normalize plane normal
        normal=normal/np.linalg.norm(normal)

        ###project points onto regression plane
        #https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d
        # curve_xy = curve_centered - np.dot(curve_centered-p_centered,normal)*normal

        curve_xy = rodrigues_rot(curve_centered, normal, [0,0,1])
        p_temp = rodrigues_rot(p_centered, normal, [0,0,1])
        p_temp = p_temp.flatten()


        xc, yc, r = fit_circle_2d(curve_xy[:,0], curve_xy[:,1],p_temp)

        ###convert to 3D coordinates
        C = rodrigues_rot(np.array([xc,yc,0]), [0,0,1], normal) + curve_mean
        C = C.flatten()
        ###get 3D circular arc
        ###always start from constraint p
        u=p-C
        if np.linalg.norm(p-curve[0])<np.linalg.norm(p-curve[-1]):
            v=curve[-1] - C
        else:
            v=curve[0] - C

    else:
        ###fit on a plane first
        curve_mean = curve.mean(axis=0)
        curve_centered = curve - curve_mean
        U,s,V = np.linalg.svd(curve_centered)
        # Normal vector of fitting plane is given by 3rd column in V
        # Note linalg.svd returns V^T, so we need to select 3rd row from V^T
        normal = V[2,:]

        curve_xy = rodrigues_rot(curve_centered, normal, [0,0,1])
        xc, yc, r = fit_circle_2d(curve_xy[:,0], curve_xy[:,1])

        ###convert to 3D coordinates
        C = rodrigues_rot(np.array([xc,yc,0]), [0,0,1], normal) + curve_mean
        C = C.flatten()
        ###get 3D circular arc
        u = curve[0] - C
        v = curve[-1] - C
   
    

    theta = angle_between(u, v, normal)

    l = np.linspace(0, theta, 1000)
    curve_fitarc = generate_circle_by_vectors(l, C, r, normal, u)
    l = np.linspace(0, 2*np.pi, 1000)
    curve_fitcircle = generate_circle_by_vectors(l, C, r, normal, u)

    return curve_fitarc, curve_fitcircle