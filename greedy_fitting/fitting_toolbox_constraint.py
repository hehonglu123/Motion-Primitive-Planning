import time

import numpy as np
import sys, copy

sys.path.append('../toolbox')
from toolbox_circular_fit import *
from lambda_calc import *
from scipy.optimize import lsq_linear

import cvxpy as cp
import gurobipy as gp


class fitting_toolbox(object):
    def __init__(self, robot, curve_js, curve=[]):
        ###robot: robot class
        ###curve_js: points in joint space
        ###d: standoff distance
        self.curve_js = curve_js
        self.robot = robot

        ###get full orientation list
        if len(curve) > 0:
            self.curve_R = []
            self.curve = curve
            for i in range(len(curve_js)):
                pose_temp = self.robot.fwd(curve_js[i])
                self.curve_R.append(pose_temp.R)
        else:
            self.curve_R = []
            self.curve = []
            for i in range(len(curve_js)):
                pose_temp = self.robot.fwd(curve_js[i])
                self.curve_R.append(pose_temp.R)
                self.curve.append(pose_temp.p)

        self.curve_R = np.array(self.curve_R)
        self.curve = np.array(self.curve)

        self.lam = calc_lam_cs(self.curve)

        ###seed initial js for inv
        self.q_prev = curve_js[0]

        self.curve_fit = []
        self.curve_fit_R = []
        self.curve_fit_js = []
        self.cartesian_slope_prev = None
        self.js_slope_prev = None

    def R2w(self, curve_R, R_constraint=[]):
        if len(R_constraint) == 0:
            R_init = curve_R[0]
            curve_w = [np.zeros(3)]
        else:
            R_init = R_constraint
            R_diff = np.dot(curve_R[0], R_init.T)
            k, theta = R2rot(R_diff)
            k = np.array(k)
            curve_w = [k * theta]

        for i in range(1, len(curve_R)):
            R_diff = np.dot(curve_R[i], R_init.T)
            k, theta = R2rot(R_diff)
            k = np.array(k)
            curve_w.append(k * theta)
        return np.array(curve_w)

    def w2R(self, curve_w, R_init):
        curve_R = []
        for i in range(len(curve_w)):
            theta = np.linalg.norm(curve_w[i])
            if theta == 0:
                curve_R.append(R_init)
            else:
                curve_R.append(np.dot(rot(curve_w[i] / theta, theta), R_init))

        return np.array(curve_R)

    def linear_interp(self, p_start, p_end, steps):
        slope = (p_end - p_start) / (steps - 1)
        return np.dot(np.arange(0, steps).reshape(-1, 1), slope.reshape(1, -1)) + p_start

    def orientation_interp(self, R_init, R_end, steps):
        curve_fit_R = []
        ###find axis angle first
        R_diff = np.dot(R_init.T, R_end)
        k, theta = R2rot(R_diff)
        for i in range(steps):
            ###linearly interpolate angle
            angle = theta * i / (steps - 1)
            R = rot(k, angle)
            curve_fit_R.append(np.dot(R_init, R))
        curve_fit_R = np.array(curve_fit_R)
        return curve_fit_R

    def car2js(self, curve_fit, curve_fit_R):

        ###calculate corresponding joint configs
        curve_fit_js = []
        if curve_fit.shape == (3,):
            q_all = np.array(self.robot.inv(curve_fit, curve_fit_R))

            ###choose inv_kin closest to previous joints
            if len(self.curve_fit_js) > 1:
                temp_q = q_all - self.curve_fit_js[-1]
            else:
                temp_q = q_all - self.curve_js[0]
            order = np.argsort(np.linalg.norm(temp_q, axis=1))
            curve_fit_js.append(q_all[order[0]])
        else:
            for i in range(len(curve_fit)):
                q_all = np.array(self.robot.inv(curve_fit[i], curve_fit_R[i]))

                ###choose inv_kin closest to previous joints
                if len(self.curve_fit_js) > 1:
                    temp_q = q_all - self.curve_fit_js[-1]
                else:
                    temp_q = q_all - self.curve_js[0]
                order = np.argsort(np.linalg.norm(temp_q, axis=1))
                curve_fit_js.append(q_all[order[0]])
        return curve_fit_js

    def quatera(self, curve_quat, initial_quat=[]):
        ###quaternion regression
        if len(initial_quat) == 0:
            Q = np.array(curve_quat).T
            Z = np.dot(Q, Q.T)
            u, s, vh = np.linalg.svd(Z)

            w = np.dot(quatproduct(u[:, 1]), quatcomplement(u[:, 0]))
            k, theta = q2rot(w)  # get the axis of rotation

            theta1 = 2 * np.arctan2(np.dot(u[:, 1], curve_quat[0]), np.dot(u[:, 0], curve_quat[0]))
            theta2 = 2 * np.arctan2(np.dot(u[:, 1], curve_quat[-1]), np.dot(u[:, 0], curve_quat[-1]))

            # get the angle of rotation
            theta = (theta2 - theta1) % (2 * np.pi)
            if theta > np.pi:
                theta -= 2 * np.pi

        else:
            ###TODO: find better way for orientation continuous constraint
            curve_quat_cons = np.vstack((curve_quat, np.tile(initial_quat, (999999, 1))))
            Q = np.array(curve_quat_cons).T
            Z = np.dot(Q, Q.T)
            u, s, vh = np.linalg.svd(Z)

            w = np.dot(quatproduct(u[:, 1]), quatcomplement(u[:, 0]))
            k, theta = q2rot(w)

            theta1 = 2 * np.arctan2(np.dot(u[:, 1], curve_quat[0]), np.dot(u[:, 0], curve_quat[0]))
            theta2 = 2 * np.arctan2(np.dot(u[:, 1], curve_quat[-1]), np.dot(u[:, 0], curve_quat[-1]))

            # get the angle of rotation
            theta = (theta2 - theta1) % (2 * np.pi)
            if theta > np.pi:
                theta -= 2 * np.pi

        curve_fit_R = []
        R_init = q2R(curve_quat[0])

        for i in range(len(curve_quat)):
            ###linearly interpolate angle
            angle = theta * i / len(curve_quat)
            R = rot(k, angle)
            curve_fit_R.append(np.dot(R, R_init))
        curve_fit_R = np.array(curve_fit_R)
        return curve_fit_R

    def threshold_slope(self, slope_prev, slope, slope_thresh):
        slope_norm = np.linalg.norm(slope)
        slope_prev = slope_prev / np.linalg.norm(slope_prev)
        slope = slope.flatten() / slope_norm

        angle = np.arccos(np.dot(slope_prev, slope))

        if abs(angle) > slope_thresh:
            slope_ratio = np.sin(slope_thresh) / np.sin(abs(angle) - slope_thresh)
            slope_new = slope_prev + slope_ratio * slope
            slope_new = slope_norm * slope_new / np.linalg.norm(slope_new)

            return slope_new / np.linalg.norm(slope_new)

        else:
            return slope

    def constrained_lstsq(self, A, b, lb, ub, slope_prev, slope_next):
        slope_prev = np.zeros(b.shape[1]) if len(slope_prev) == 0 else slope_prev
        slope_next = np.zeros(b.shape[1]) if len(slope_next) == 0 else slope_next
        jac = self.robot.jacobian(self.curve_fit_js[-1])
        inv_jac = np.linalg.pinv(jac)
        # x_sol = np.zeros((A.shape[1], b.shape[1]))
        # for i in range(b.shape[1]):
        # 	x = cp.Variable(A.shape[1])
        # 	objective = cp.Minimize(cp.sum(A @ x - b[:, i]))
        # 	constraints = [inv_jac[]@(x - slope_prev[i]) >= lb[i], inv_jac@(x - slope_prev[i]) <= ub[i]]
        # 	prob = cp.Problem(objective, constraints)
        # 	prob.solve()
        # 	x_sol[i] = x.value
        # return x_sol
        x = cp.Variable((A.shape[1], b.shape[1]))
        objective = cp.Minimize(cp.norm(A @ x - b, 'fro'))
        constraints = [inv_jac @ (x[-1, :] - slope_prev).reshape(-1, 1) >= lb, inv_jac @ (x[-1, :] - slope_prev).reshape(-1, 1) <= ub,
                       inv_jac @ (x[-1, :] - slope_next).reshape(-1, 1) >= lb, inv_jac @ (x[-1, :] - slope_next).reshape(-1, 1) <= ub]
        prob = cp.Problem(objective, constraints)
        # result = prob.solve(solver=cp.SCS, verbose=False, max_iters=1000, use_indirect=True)
        prob.solve(solver='GUROBI', verbose=True)
        # if x.value is None:
        # 	print(result)
        return x.value

    def constrained_lstsq_js(self, A, b, lb, ub, slope_prev, slope_next):
        slope_prev = np.zeros(b.shape[1]) if len(slope_prev) == 0 else slope_prev
        slope_next = np.zeros(b.shape[1]) if len(slope_next) == 0 else slope_next
        x = cp.Variable((A.shape[1], b.shape[1]))
        objective = cp.Minimize(cp.norm(A @ x - b, 'fro'))
        constraints = [x[-1, :] - slope_prev >= lb, x[-1, :] - slope_prev <= ub,
                       x[-1, :] - slope_next >= lb, x[-1, :] - slope_next <= ub]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver='GUROBI', verbose=False)
        return x.value

    # A = np.hstack([A for _ in range(b.shape[1])])
    # model = gp.Model('lstsq')
    # x = model.addMVar(shape=(A.shape[1], b.shape[1]), name='x')
    # model.setObjective()
    # model.addMConstr(A=np.identity(b.shape[1]), x=x, sense='>', b=lb + slope_prev)
    # model.addMConstr(A=np.identity(b.shape[1]), x=x, sense='<', b=ub + slope_prev)
    # model.optimize()
    # x_sol = model.getVarByName('x').X
    # print(x_sol)
    # return x_sol

    # x_sol = np.zeros((A.shape[1], b.shape[1]))
    # for i in range(b.shape[1]):
    # 	x = cp.Variable(A.shape[1])
    # 	objective = cp.Minimize(cp.sum_squares(A @ x - b[:, i]))
    # 	constraints = [(x - slope_prev[i]) >= lb[:, i], (x - slope_prev[i]) <= ub[:, i]]
    # 	prob = cp.Problem(objective, constraints)
    # 	prob.solve()
    # 	x_sol[:, i] = x.value
    # return x_sol

    def linear_fit(self, data, p_constraint=[], slope_prev=[], slope_next=[], key='L'):
        slope_lb = - np.ones(data.shape[1]) * np.inf
        slope_ub = np.ones(data.shape[1]) * np.inf
        if len(slope_prev) > 0 or len(slope_next) > 0:
            if key == 'L':
                slope_lb = np.ones(data.shape[1]) * np.tan(- np.pi / 18) * 0.01
                slope_ub = np.ones_likes(data.shape[1]) * np.tan(np.pi / 18) * 0.01
            else:
                slope_lb = np.ones(data.shape[1]) * np.tan(- np.pi / 72)
                slope_ub = np.ones(data.shape[1]) * np.tan(np.pi / 72)
        # if key == 'L':
        # 	for i in range(data.shape[1]):
        # 		slope_lb[i] = np.tan(- np.pi / 36)
        # 		slope_ub[i] = np.tan(np.pi / 36)
        # else:
        # 	for i in range(data.shape[1]):
        # 		slope_lb[i] = np.tan(np.arctan(slope_prev[i]) - np.pi / 36)
        # 		slope_ub[i] = np.tan(np.arctan(slope_prev[i]) + np.pi / 36)

        ###no constraint
        if len(p_constraint) == 0:
            A = np.vstack((np.ones(len(data)), np.arange(0, len(data)))).T
            b = data
            if key == 'L':
                res_opt = self.constrained_lstsq(A, b, slope_lb, slope_ub, slope_prev, slope_next)
            elif key == 'J':
                res_opt = self.constrained_lstsq_js(A, b, slope_lb, slope_ub, slope_prev, slope_next)
            else:
                res_opt = np.linalg.lstsq(A, b, rcond=None)[0]
            # res_opt=np.linalg.lstsq(A,b,rcond=None)[0]
            if res_opt is None:
                return None
            start_point = res_opt[0]
            slope = res_opt[1].reshape(1, -1)

            data_fit = np.dot(np.arange(0, len(data)).reshape(-1, 1), slope) + start_point
        ###with constraint point
        else:
            start_point = p_constraint

            A = np.arange(1, len(data) + 1).reshape(-1, 1)
            b = data - start_point
            if key == 'L':
                res_opt = self.constrained_lstsq(A, b, slope_lb, slope_ub, slope_prev, slope_next)
            elif key == 'J':
                # res_opt = np.zeros((1, data.shape[1]))
                # for i in range(data.shape[1]):
                # 	slope_bounds = (slope_lb[i], slope_ub[i])
                # 	lsq_res = lsq_linear(A, b[:, i], bounds=slope_bounds)
                # 	res_opt[:, i] = lsq_res.x
                res_opt = self.constrained_lstsq_js(A, b, slope_lb, slope_ub, slope_prev, slope_next)
            else:
                res_opt = np.linalg.lstsq(A, b, rcond=None)[0]
            # res_opt=np.linalg.lstsq(A,b,rcond=None)[0]
            if res_opt is None:
                return None
            slope = res_opt.reshape(1, -1)
            # violation_low, violation_high = self.constrain_violation(slope, slope_lb, slope_ub, slope_prev, key)
            # if violation_low > 0 or violation_high > 0:
            # 	print("Violate Slope Constrain: Move{} - ({:.5f}, {:.5f})".format(key, violation_low, violation_high))

            data_fit = np.dot(np.arange(1, len(data) + 1).reshape(-1, 1), slope) + start_point

        return data_fit

    def get_start_slope(self, p1, p2, R1, R2):
        q1 = self.car2js(p1, R1)[0]
        q2 = self.car2js(p2, R2)[0]

        return (q2 - q1) / np.linalg.norm(q2 - q1)

    def movel_fit(self, curve, curve_js, curve_R, p_constraint=[], R_constraint=[], slope_prev=[],
                  slope_next=[]):  ###unit vector slope
        ###convert orientation to w first
        curve_w = self.R2w(curve_R, R_constraint)

        curve_slope = self.curve_fit[-1] - self.curve_fit[-2] if len(self.curve_fit) > 1 else []
        curve_fit_w = self.R2w(self.curve_fit_R[-2:]) if len(self.curve_fit_R) > 1 else []
        curve_w_slope = curve_fit_w[-1] - curve_fit_w[-2] if len(curve_fit_w) > 1 else []
        cartesian_slope = np.hstack([curve_slope, curve_w_slope]) if len(self.curve_fit) > 1 else []
        data_fit = self.linear_fit(np.hstack((curve, curve_w)),
                                   [] if len(p_constraint) == 0 else np.hstack((p_constraint, np.zeros(3))),
                                   slope_prev=cartesian_slope, key='L')
        if data_fit is None:
            return [], [], [], 999, 999
        curve_fit = data_fit[:, :3]
        curve_fit_w = data_fit[:, 3:]

        curve_fit_R = self.w2R(curve_fit_w, curve_R[0] if len(R_constraint) == 0 else R_constraint)

        p_error = np.linalg.norm(curve - curve_fit, axis=1)

        curve_fit_R = np.array(curve_fit_R)
        ori_error = []
        for i in range(len(curve)):
            ori_error.append(get_angle(curve_R[i, :, -1], curve_fit_R[i, :, -1]))

        ###slope thresholding
        if len(slope_prev) > 0:
            slope_cur = self.get_start_slope(self.curve_fit[-1], curve_fit[0], self.curve_fit_R[-1], curve_fit_R[0])
            if get_angle(slope_cur, slope_prev) > self.slope_constraint:
                # print('triggering slope threshold')
                slope_new = self.threshold_slope(slope_prev, slope_cur, self.slope_constraint)
                ###propogate to L slope
                J = self.robot.jacobian(self.curve_fit_js[-1])
                nu = J @ slope_new
                p_slope = nu[3:] / np.linalg.norm(nu[3:])
                w_slope = nu[:3] / np.linalg.norm(nu[:3])
                ###find correct length
                lam_p = p_slope @ (curve[-1] - p_constraint)
                lam_p_all = np.linspace(0, lam_p, num=len(curve) + 1)[1:].reshape(-1, 1)
                lam_w = w_slope @ (curve_w[-1])
                lam_w_all = np.linspace(0, lam_w, num=len(curve) + 1)[1:].reshape(-1, 1)
                # position

                curve_fit = lam_p_all @ p_slope.reshape(1, -1) + p_constraint
                p_error = np.linalg.norm(curve - curve_fit, axis=1)
                # orientation
                curve_fit_w = lam_w_all @ w_slope.reshape(1, -1)
                curve_fit_R = self.w2R(curve_fit_w, R_constraint)
                ori_error = []
                for i in range(len(curve)):
                    ori_error.append(get_angle(curve_R[i, :, -1], curve_fit_R[i, :, -1]))

        return curve_fit, curve_fit_R, [], np.max(p_error), np.max(ori_error)

    def movej_fit(self, curve, curve_js, curve_R, p_constraint=[], R_constraint=[], slope_prev=[], slope_next=[]):
        ###convert orientation to w first
        curve_w = self.R2w(curve_R, R_constraint)

        curve_fit_js = self.linear_fit(curve_js, p_constraint, slope_prev=slope_prev, slope_next=slope_next, key='J')
        if curve_fit_js is None:
            print(slope_prev, slope_next)
            return [], [], [], 999, 999

        ###necessary to fwd every point search to get error calculation
        curve_fit = []
        curve_fit_R = []
        try:
            for i in range(len(curve_fit_js)):
                pose_temp = self.robot.fwd(curve_fit_js[i])
                curve_fit.append(pose_temp.p)
                curve_fit_R.append(pose_temp.R)
        except:
            print("Invalid fwd")
            return curve_fit, curve_fit_R, curve_fit_js, 999, 999
        else:
            curve_fit = np.array(curve_fit)
            curve_fit_R = np.array(curve_fit_R)

            ###error
            p_error = np.linalg.norm(curve - curve_fit, axis=1)
            curve_fit_R = np.array(curve_fit_R)
            ori_error = []
            for i in range(len(curve)):
                ori_error.append(get_angle(curve_R[i, :, -1], curve_fit_R[i, :, -1]))

            ###slope thresholding
            if len(slope_prev) > 0:
                # print('triggering slope threshold')
                slope_cur = self.get_start_slope(self.curve_fit[-1], curve_fit[0], self.curve_fit_R[-1], curve_fit_R[0])
                if get_angle(slope_cur, slope_prev) > self.slope_constraint:
                    slope_new = self.threshold_slope(slope_prev, slope_cur, self.slope_constraint)
                    ###find correct length
                    lam_js = slope_new @ (curve_js[-1] - p_constraint)
                    lam_js_all = np.linspace(0, lam_js, num=len(curve) + 1)[1:].reshape(-1, 1)
                    # joints propogation
                    curve_fit_js = lam_js_all @ slope_new.reshape(1, -1) + p_constraint
                    curve_fit = []
                    curve_fit_R = []
                    for i in range(len(curve_fit_js)):
                        pose_temp = self.robot.fwd(curve_fit_js[i])
                        curve_fit.append(pose_temp.p)
                        curve_fit_R.append(pose_temp.R)
                    curve_fit = np.array(curve_fit)
                    curve_fit_R = np.array(curve_fit_R)

                    ###error
                    p_error = np.linalg.norm(curve - curve_fit, axis=1)
                    curve_fit_R = np.array(curve_fit_R)
                    ori_error = []
                    for i in range(len(curve)):
                        ori_error.append(get_angle(curve_R[i, :, -1], curve_fit_R[i, :, -1]))
            return curve_fit, curve_fit_R, curve_fit_js, np.max(p_error), np.max(ori_error)

    def movec_fit(self, curve, curve_js, curve_R, p_constraint=[], R_constraint=[], slope_prev=[], slope_next=[]):
        curve_w = self.R2w(curve_R, R_constraint)

        curve_fit, curve_fit_circle = circle_fit(curve, [] if len(R_constraint) == 0 else p_constraint)
        curve_fit_w = self.linear_fit(curve_w, [] if len(R_constraint) == 0 else np.zeros(3), key='C')

        curve_fit_R = self.w2R(curve_fit_w, curve_R[0] if len(R_constraint) == 0 else R_constraint)

        p_error = np.linalg.norm(curve - curve_fit, axis=1)

        curve_fit_R = np.array(curve_fit_R)
        ori_error = []
        for i in range(len(curve)):
            ori_error.append(get_angle(curve_R[i, :, -1], curve_fit_R[i, :, -1]))

        ###slope thresholding
        if len(slope_prev) > 0:
            # print('triggering slope threshold')
            slope_cur = self.get_start_slope(self.curve_fit[-1], curve_fit[0], self.curve_fit_R[-1], curve_fit_R[0])
            if get_angle(slope_cur, slope_prev) > self.slope_constraint:
                slope_new = self.threshold_slope(slope_prev, slope_cur, self.slope_constraint)
                ###propogate to L slope
                J = self.robot.jacobian(self.curve_fit_js[-1])
                nu = J @ slope_new
                p_slope = nu[3:] / np.linalg.norm(nu[3:])
                w_slope = nu[:3] / np.linalg.norm(nu[:3])
                ###position
                curve_fit, curve_fit_circle = circle_fit_w_slope1(curve, p_constraint, p_slope)
                p_error = np.linalg.norm(curve - curve_fit, axis=1)
                lam_w = w_slope @ (curve_w[-1])
                lam_w_all = np.linspace(0, lam_w, num=len(curve) + 1)[1:].reshape(-1, 1)
                ###orientation
                curve_fit_w = lam_w_all @ w_slope.reshape(1, -1)
                curve_fit_R = self.w2R(curve_fit_w, R_constraint)
                ori_error = []
                for i in range(len(curve)):
                    ori_error.append(get_angle(curve_R[i, :, -1], curve_fit_R[i, :, -1]))

        return curve_fit, curve_fit_R, [], np.max(p_error), np.max(ori_error)

    def get_slope(self, curve_fit, curve_fit_R, breakpoints):
        slope_diff = []
        slope_diff_ori = []
        for i in range(1, len(breakpoints) - 1):
            slope_diff.append(get_angle(curve_fit[breakpoints[i] - 1] - curve_fit[breakpoints[i] - 2],
                                        curve_fit[breakpoints[i]] - curve_fit[breakpoints[i] - 1]))

            R_diff_prev = np.dot(curve_fit_R[breakpoints[i]], curve_fit_R[breakpoints[i - 1]].T)
            k_prev, theta = R2rot(R_diff_prev)
            R_diff_next = np.dot(curve_fit_R[breakpoints[i + 1] - 1], curve_fit_R[breakpoints[i]].T)
            k_next, theta = R2rot(R_diff_next)
            slope_diff_ori.append(get_angle(k_prev, k_next, less90=True))

        return slope_diff, slope_diff_ori

    def get_slope_js(self, curve_fit_js, breakpoints):
        slope_diff_js = []

        for i in range(1, len(breakpoints) - 1):
            slope1 = curve_fit_js[breakpoints[i] - 1] - curve_fit_js[breakpoints[i] - 2]
            slope2 = curve_fit_js[breakpoints[i]] - curve_fit_js[breakpoints[i] - 1]
            slope_diff = np.abs(np.arctan(slope1) - np.arctan(slope2))
            slope_diff = np.degrees(slope_diff)
            slope_diff = ["{:.3f}".format(slope_diff[i]) for i in range(len(slope_diff))]
            slope_diff_js.append("\t".join(slope_diff))
        ret = "\n".join(slope_diff_js)

        return ret


def main():
    ###read in points
    col_names = ['X', 'Y', 'Z', 'direction_x', 'direction_y', 'direction_z']
    data = read_csv("../data/from_cad/Curve_in_base_frame.csv", names=col_names)
    curve_x = data['X'].tolist()
    curve_y = data['Y'].tolist()
    curve_z = data['Z'].tolist()
    curve = np.vstack((curve_x, curve_y, curve_z)).T

    curve_fit, max_error_all = fit_w_breakpoints(curve, [movel_fit, movec_fit, movec_fit],
                                                 [0, int(len(curve) / 3), int(2 * len(curve) / 3), len(curve)])

    print(max_error_all)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(curve[:, 0], curve[:, 1], curve[:, 2], 'gray')
    ax.plot3D(curve_fit[:, 0], curve_fit[:, 1], curve_fit[:, 2], 'red')

    plt.show()


if __name__ == "__main__":
    main()
