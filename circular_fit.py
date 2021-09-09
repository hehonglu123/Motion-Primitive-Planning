from numpy import empty, sqrt, square
from scipy.linalg import lstsq
from pandas import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def nsphere_fit(x, axis=-1, scaling=False):
    r"""
    Fit an n-sphere to ND data.
    The center and radius of the n-sphere are optimized using the Coope
    method. The sphere is described by
    .. math::
       \left \lVert \vec{x} - \vec{c} \right \rVert_2 = r
    Parameters
    ----------
    x : array-like
        The n-vectors describing the data. Usually this will be a nxm
        array containing m n-dimensional data points.
    axis : int
        The axis that determines the number of dimensions of the
        n-sphere. All other axes are effectively raveled to obtain an
        ``(m, n)`` array.
    scaling : bool
        If `True`, scale and offset the data to a bounding box of -1 to
        +1 during computations for numerical stability. Default is
        `False`.
    Return
    ------
    r : scalar
        The optimal radius of the best-fit n-sphere for `x`.
    c : array
        An array of size `x.shape[axis]` with the optimized center of
        the best-fit n-sphere.
    References
    ----------
    - [Coope]_ "\ :ref:`ref-cfblanls`\ "
    """
    n = x.shape[-1]
    x = x.reshape(-1, n)
    m = x.shape[0]

    B = empty((m, n + 1), dtype=x.dtype)
    X = B[:, :-1]
    X[:] = x
    B[:, -1] = 1

    if scaling:
        xmin = X.min()
        xmax = X.max()
        scale = 0.5 * (xmax - xmin)
        offset = 0.5 * (xmax + xmin)
        X -= offset
        X /= scale

    d = square(X).sum(axis=-1)

    y, *_ = lstsq(B, d, overwrite_a=True, overwrite_b=True)

    c = 0.5 * y[:-1]
    r = sqrt(y[-1] + square(c).sum())

    if scaling:
        r *= scale
        c *= scale
        c += offset

    return r, c


col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
data = read_csv("data/from_interp/Curve_in_base_frame.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve=np.vstack((curve_x, curve_y, curve_z)).T
r,c=nsphere_fit(curve)


###show sphere
plt.figure()
ax = plt.axes(projection='3d')

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = r*np.cos(u)*np.sin(v)+c[0]
y = r*np.sin(u)*np.sin(v)+c[1]
z = r*np.cos(v)+c[2]
ax.plot_wireframe(x, y, z, color="r")



ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'gray')
ax.scatter3D(curve[:,0], curve[:,1], curve[:,2], c=curve[:,2], cmap='Greens')
# ax.scatter3D(curve_cartesian_pred[:,0], curve_cartesian_pred[:,1], curve_cartesian_pred[:,2], c=curve_cartesian_pred[:,2], cmap='Blues')
plt.show()


