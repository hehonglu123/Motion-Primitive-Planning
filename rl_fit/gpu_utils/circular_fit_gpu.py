import numpy as np
import traceback
from scipy.optimize import minimize

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
print(device)


def generate_circle_by_vectors(t, C, r, n, u):
    n = n / torch.norm(n)
    u = u / torch.norm(u)
    P_circle = r * torch.cos(t)[:, None] * u + r * torch.sin(t)[:, None] * torch.cross(n, u) + C
    return P_circle


def fit_circle_2d(x, y, p=[], p2=[]):
    if len(p) == 0:
        A = torch.as_tensor([x, y, torch.ones(len(x))]).T
        b = x ** 2 + y ** 2

        # Solve by method of least squares
        c = torch.lstsq(A, b).solution

        # Get circle parameters from solution c
        xc = c[0] / 2
        yc = c[1] / 2
        r = torch.sqrt(c[2] + xc ** 2 + yc ** 2)
        return xc, yc, r
    elif len(p2) == 0:

        ###rewrite lstsq to fit point p on circle
        A = torch.stack([x - p[0], y - p[1]]).T
        b = x ** 2 + y ** 2 - p[0] ** 2 - p[1] ** 2

        # Solve by method of least squares
        c = torch.linalg.lstsq(A, b).solution

        # Get circle parameters from solution c
        xc = c[0] / 2
        yc = c[1] / 2
        r = torch.norm(p[:-1] - torch.as_tensor([xc, yc], device=device))
        return xc, yc, r
    else:
        A_x = (p[0] + p2[0]) / 2
        A_y = (p[1] + p2[1]) / 2
        vT = torch.as_tensor([p[1] - p2[1], p2[0] - p[0]])
        vT = vT / torch.norm(vT)
        A = torch.as_tensor([vT[0] * (x - p[0]) + vT[1] * (y - p[1])]).T
        b = x ** 2 + y ** 2 - p[0] ** 2 - p[1] ** 2 - 2 * A_x * x + 2 * A_x * p[0] - 2 * A_y * y + 2 * A_y * p[1]
        d = torch.linalg.lstsq(A, b).solution
        xc = A_x + d * vT[0]
        yc = A_y + d * vT[1]
        r = abs(d)

        return xc, yc, r


def fit_circle_2d_w_slope(x, curve, p):
    # x:0-2 direction, 3 radius
    center = p - x[-1] * x[:3]
    return torch.norm(
        x[-1] ** 2 - torch.norm(curve - center, axis=1))  ###min{ || (x-x_c)^2+(y-y_c)^2+(z-z_c)^2 - r^2 ||  }


def fit_circle_2d_w_slope2(x, curve, p, r_dir):
    # given direction already
    center = p - x * r_dir
    return torch.norm(
        x ** 2 - torch.norm(curve - center, axis=1) ** 2)  ###min{ || (x-x_c)^2+(y-y_c)^2+(z-z_c)^2 - r^2 ||  }


def vec_proj_plane(u, n):
    ###u: vector in 3D
    ###n: plane normal
    proj_of_u_on_n = u - (torch.dot(u, n) / n ** 2) * n

    return proj_of_u_on_n


# -------------------------------------------------------------------------------
# RODRIGUES ROTATION
# - Rotate given points based on a starting and ending vector
# - Axis k and angle of rotation theta given by vectors n0,n1
#   curve_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))
# -------------------------------------------------------------------------------
def rodrigues_rot(curve, n0, n1):
    # If curve is only 1d array (coords of single point), fix it to be matrix
    if curve.ndim == 1:
        curve = curve[None, :]

    # Get vector of rotation k and angle theta
    n0 = n0 / torch.norm(n0)
    n1 = n1 / torch.norm(n1)
    k = torch.cross(n0.float(), n1.float())
    k = k / torch.norm(k)
    theta = torch.arccos(torch.dot(n0.float(), n1.float()))

    # Compute rotated points
    curve_rot = torch.zeros((len(curve), 3), device=device)
    curve = curve.float()
    for i in range(len(curve)):
        curve_rot[i] = curve[i] * torch.cos(theta) + torch.cross(k, curve[i]) * torch.sin(theta) + \
                       k * torch.dot(k, curve[i]) * (1 - torch.cos(theta))

    return curve_rot


# -------------------------------------------------------------------------------
# ANGLE BETWEEN
# - Get angle between vectors u,v with sign based on plane with unit normal n
# -------------------------------------------------------------------------------
def angle_between(u, v, n=None):
    if n is None:
        return torch.arctan2(torch.norm(torch.cross(u, v)), torch.dot(u, v))
    else:
        return torch.arctan2(torch.dot(n, torch.cross(u, v)), torch.dot(u, v))


# -------------------------------------------------------------------------------
# - Make axes of 3D plot to have equal scales
# - This is a workaround to Matplotlib's set_aspect('equal') and axis('equal')
#   which were not working for 3D
# -------------------------------------------------------------------------------
def set_axes_equal_3d(ax):
    limits = torch.as_tensor([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = torch.abs(limits[:, 0] - limits[:, 1])
    centers = torch.mean(limits, dim=1)
    radius = 0.5 * torch.max(spans)
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])

    ###fit curve with slope and 1 p constraint


def circle_fit_w_slope1(curve, p, slope):
    ###fit on a plane first
    curve_mean = curve.mean(dim=0)
    curve_centered = curve - curve_mean
    p_centered = p - curve_mean

    ###constraint fitting
    ###rewrite lstsq to fit point p on plane
    A = torch.as_tensor([curve_centered[:, 0] - p_centered[0] * curve_centered[:, 2] / p_centered[2],
                  curve_centered[:, 1] - p_centered[1] * curve_centered[:, 2] / p_centered[2]]).T
    b = torch.ones(len(curve)) - curve_centered[:, 2] / p_centered[2]
    c = torch.lstsq(A, b).solution
    normal = torch.as_tensor([c[0], c[1], (1 - c[0] * p_centered[0] - c[1] * p_centered[1]) / p_centered[2]])

    ###make sure constraint point is on plane
    # print(np.dot(normal,p_centered))
    ###normalize plane normal
    normal = normal / torch.norm(normal)

    ###find the line direction where the center of the circle reside
    r_dir = torch.cross(slope, normal)
    r_dir = r_dir / torch.norm(r_dir)

    circle_plane_normal = torch.cross(r_dir, slope)
    circle_plane_normal = circle_plane_normal / torch.norm(circle_plane_normal)

    res = minimize(fit_circle_2d_w_slope2, [5000], method='SLSQP', tol=1e-10, args=(curve, p, r_dir,))  # TODO: Check this line
    # print('radius: ',res.x)
    r = abs(res.x)
    C = p - res.x * r_dir
    end_vec = vec_proj_plane(curve[-1] - C, circle_plane_normal)

    ###get 3D circular arc
    u = p - C
    if torch.norm(p - curve[0]) < torch.norm(p - curve[-1]):
        v = curve[-1] - C
    else:
        v = curve[0] - C
    theta = angle_between(u, v, circle_plane_normal).data

    l = torch.linspace(0, theta, len(curve))
    curve_fitarc = generate_circle_by_vectors(l, C, r, circle_plane_normal, u)
    l = torch.linspace(0, 2 * np.pi, 1000)
    curve_fitcircle = generate_circle_by_vectors(l, C, r, circle_plane_normal, u)

    return curve_fitarc, curve_fitcircle


def get_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = torch.as_tensor([a1, a2, b1, b2])  # s for stacked
    h = torch.hstack((s, torch.ones((4, 1))))  # h for homogeneous
    l1 = torch.cross(h[0], h[1])  # get first line
    l2 = torch.cross(h[2], h[3])  # get second line
    x, y, z = torch.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return (float('inf'), float('inf'))
    return (x / z, y / z)


def circle_fit_w_2slope(curve, p, p2, slope1, slope2):
    ###fit a circle with 2 points constraint and 2 slope constraints
    ###fit a plane with 2 point constraint
    curve_mean = curve.mean(dim=0)
    curve_centered = curve - curve_mean
    p_centered = p - curve_mean
    p2_centered = p2 - curve_mean

    ###constraint fitting
    ###rewrite lstsq to fit point p on plane
    A = torch.as_tensor([curve_centered[:, 0] - p_centered[0] * curve_centered[:, 2] / p_centered[2] - (
                p2_centered[0] - p_centered[0] * p2_centered[2] / p_centered[2]) * (
                              curve_centered[:, 1] - p_centered[1] * curve_centered[:, 2] / p_centered[2]) / (
                              p2_centered[1] - p_centered[1] * p2_centered[2] / p_centered[2])]).T
    b = torch.ones(curve.shape[0], device=device) - curve_centered[:, 2] / p_centered[2] - (1 - p2_centered[2] / p_centered[2]) * (
                curve_centered[:, 1] - p_centered[1] * curve_centered[:, 2] / p_centered[2]) / (
                    p2_centered[1] - p_centered[1] * p2_centered[2] / p_centered[2])

    c = torch.lstsq(A, b).solution
    A_out = c[0]
    B_out = (1 - p2_centered[2] / p_centered[2] - A_out * (
                p2_centered[0] - p_centered[0] * p2_centered[2] / p_centered[2])) / (
                        p2_centered[1] - p_centered[1] * p2_centered[2] / p_centered[2])
    normal = torch.as_tensor([A_out, B_out, (1 - A_out * p_centered[0] - B_out * p_centered[1]) / p_centered[2]])

    ###make sure constraint point is on plane
    # print(np.dot(normal,p_centered))
    # print(np.dot(normal,p2_centered))
    ###normalize plane normal
    normal = normal / torch.norm(normal)

    curve_xy = rodrigues_rot(curve_centered, normal, [0, 0, 1])
    p_temp1 = rodrigues_rot(p_centered, normal, [0, 0, 1]).flatten()[:-1]
    p_temp2 = rodrigues_rot(p2_centered, normal, [0, 0, 1]).flatten()[:-1]
    slope_temp1 = rodrigues_rot(slope1, normal, [0, 0, 1]).flatten()[:-1]
    slope_temp2 = rodrigues_rot(slope2, normal, [0, 0, 1]).flatten()[:-1]

    xc, yc = get_intersect(p_temp1, p_temp1 + np.array([-slope_temp1[1], slope_temp1[0]]), p_temp2,
                           p_temp2 + np.array([-slope_temp2[1], slope_temp2[0]]))
    r = torch.norm(p_temp1 - np.array([xc, yc]))

    ###convert to 3D coordinates
    C = rodrigues_rot(torch.as_tensor([xc, yc, 0]), [0, 0, 1], normal) + curve_mean
    C = C.flatten()
    ###get 3D circular arc
    ###always start from constraint p
    u = p - C
    v = p2 - C

    theta = angle_between(u, v, normal).data

    l = torch.linspace(0, theta, len(curve))
    curve_fitarc = generate_circle_by_vectors(l, C, r, normal, u)
    l = torch.linspace(0, 2 * torch.pi, 1000)
    curve_fitcircle = generate_circle_by_vectors(l, C, r, normal, u)

    return curve_fitarc, curve_fitcircle


def circle_fit(curve, p=None, p2=None):
    ###curve: 3D point train_data
    ###p:   constraint point of the arc
    ########################################
    ###return curve_fit: 3D point train_data

    if p is None:  # no constraint
        ###fit on a plane first
        curve_mean = curve.mean(axis=0)
        curve_centered = curve - curve_mean
        U, s, V = torch.svd(curve_centered)
        # Normal vector of fitting plane is given by 3rd column in V
        # Note linalg.svd returns V^T, so we need to select 3rd row from V^T
        normal = V[2, :]

        curve_xy = rodrigues_rot(curve_centered, normal, [0, 0, 1])
        xc, yc, r = fit_circle_2d(curve_xy[:, 0], curve_xy[:, 1])

        ###convert to 3D coordinates
        C = rodrigues_rot(torch.as_tensor([xc, yc, 0]), [0, 0, 1], normal) + curve_mean
        C = C.flatten()
        ###get 3D circular arc
        u = curve[0] - C
        v = curve[-1] - C

        theta = angle_between(u, v, normal).data
        l = torch.linspace(0, theta, len(curve), device=device)
        curve_fitarc = generate_circle_by_vectors(l, C, r, normal, u)


    elif p2 is None:  # single point constraint
        ###fit on a plane first
        curve_mean = curve.mean(dim=0)
        curve_centered = curve - curve_mean
        p_centered = p - curve_mean

        ###constraint fitting
        ###rewrite lstsq to fit point p on plane
        A = torch.stack([curve_centered[:, 0] - p_centered[0] * curve_centered[:, 2] / p_centered[2],
                      curve_centered[:, 1] - p_centered[1] * curve_centered[:, 2] / p_centered[2]]).T
        b = torch.ones(len(curve), device=device) - curve_centered[:, 2] / p_centered[2]
        c = torch.linalg.lstsq(A, b).solution
        normal = torch.as_tensor([c[0], c[1], (1 - c[0] * p_centered[0] - c[1] * p_centered[1]) / p_centered[2]],
                                 device=device)

        ###make sure constraint point is on plane
        # print(np.dot(normal,p_centered))
        ###normalize plane normal
        normal = normal / torch.norm(normal)

        curve_xy = rodrigues_rot(curve_centered, normal.float(), torch.tensor([0., 0., 1.], device=device).float())
        p_temp = rodrigues_rot(p_centered, normal.float(), torch.tensor([0., 0., 1.], device=device).float())
        p_temp = p_temp.flatten()

        xc, yc, r = fit_circle_2d(curve_xy[:, 0], curve_xy[:, 1], p_temp)

        ###convert to 3D coordinates
        C = rodrigues_rot(torch.as_tensor([xc, yc, 0], device=device), torch.tensor([0., 0., 1.], device=device).float(), normal) + curve_mean
        C = C.flatten()
        ###get 3D circular arc
        ###always start from constraint p
        u = p - C
        if torch.norm(p - curve[0]) < torch.norm(p - curve[-1]):
            v = curve[-1] - C
        else:
            v = curve[0] - C

        theta = angle_between(u, v, normal)
        l = torch.linspace(0, theta.data, len(curve) + 1, device=device)
        curve_fitarc = generate_circle_by_vectors(l, C, r, normal, u)[1:]

    else:
        ###fit a plane with 2 point constraint
        curve_mean = curve.mean(dim=0)
        curve_centered = curve - curve_mean
        p_centered = p - curve_mean
        p2_centered = p2 - curve_mean

        ###constraint fitting
        ###rewrite lstsq to fit point p on plane
        A = torch.stack([curve_centered[:, 0] - p_centered[0] * curve_centered[:, 2] / p_centered[2] - (
                    p2_centered[0] - p_centered[0] * p2_centered[2] / p_centered[2]) * (
                                  curve_centered[:, 1] - p_centered[1] * curve_centered[:, 2] / p_centered[2]) / (
                                  p2_centered[1] - p_centered[1] * p2_centered[2] / p_centered[2])]).T
        b = torch.ones(len(curve), device=device) - curve_centered[:, 2] / p_centered[2] - (1 - p2_centered[2] / p_centered[2]) * (
                    curve_centered[:, 1] - p_centered[1] * curve_centered[:, 2] / p_centered[2]) / (
                        p2_centered[1] - p_centered[1] * p2_centered[2] / p_centered[2])

        c = torch.linalg.lstsq(A, b).solution
        A_out = c[0]
        B_out = (1 - p2_centered[2] / p_centered[2] - A_out * (
                    p2_centered[0] - p_centered[0] * p2_centered[2] / p_centered[2])) / (
                            p2_centered[1] - p_centered[1] * p2_centered[2] / p_centered[2])
        normal = torch.as_tensor([A_out, B_out, (1 - A_out * p_centered[0] - B_out * p_centered[1]) / p_centered[2]],
                                 device=device)

        ###make sure constraint point is on plane
        # print(np.dot(normal,p_centered))
        # print(np.dot(normal,p2_centered))
        ###normalize plane normal
        normal = normal / torch.norm(normal)

        curve_xy = rodrigues_rot(curve_centered, normal, [0, 0, 1])
        p_temp1 = rodrigues_rot(p_centered, normal, [0, 0, 1]).flatten()
        p_temp2 = rodrigues_rot(p2_centered, normal, [0, 0, 1]).flatten()

        xc, yc, r = fit_circle_2d(curve_xy[:, 0], curve_xy[:, 1], p_temp1, p_temp2)

        ###convert to 3D coordinates
        C = rodrigues_rot(np.array([xc, yc, 0]), [0, 0, 1], normal) + curve_mean
        C = C.flatten()
        ###get 3D circular arc
        ###always start from constraint p
        u = p - C
        v = p2 - C

        theta = angle_between(u, v, normal).data
        l = torch.linspace(0, theta, len(curve) + 1)
        curve_fitarc = generate_circle_by_vectors(l, C, r, normal, u)[1:]

    l = torch.linspace(0, 2 * np.pi, 1000, device=device)
    curve_fitcircle = generate_circle_by_vectors(l, C, r, normal, u)

    return curve_fitarc, curve_fitcircle


def seg_3dfit(seg2fit, p=[]):
    curve_fitarc, curve_fit_circle = circle_fit(seg2fit, p)
    error = []
    ###check error
    for i in range(len(curve_fitarc)):
        error_temp = torch.norm(seg2fit - curve_fitarc[i], dim=1)
        idx = torch.argmin(error_temp)
        error.append(error_temp[idx])
    return curve_fitarc, max(error)


def stepwise_3dfitting(curve, breakpoints):
    if len(breakpoints) == 1:
        print("num of breakpoints must be greater than 2")
        return
    fit = []
    for i in range(len(breakpoints) - 1):
        seg2fit = curve[breakpoints[i]:breakpoints[i + 1]]

        try:
            if i == 0:

                curve_fitarc, curve_fit_circle = circle_fit(seg2fit)
                fit.append(curve_fitarc)

            else:
                curve_fitarc, curve_fit_circle = circle_fit(seg2fit, p=fit[-1][-1])
                fit.append(curve_fitarc)
        except:
            traceback.print_exc()
            print(breakpoints)

    return torch.as_tensor(fit).reshape(-1, 3)



