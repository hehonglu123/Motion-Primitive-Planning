import numpy as np
from scipy.fft import fft
from scipy.interpolate import interp1d

from sklearn.decomposition import PCA


def reduce_interpolate(curve, n_points):
    interval = len(curve) / (n_points - 1)
    new_curve = [curve[0, :]]
    break_points = [x * interval for x in range(1, n_points-1)]
    for break_point in break_points:
        left_point_idx = np.floor(break_point).astype(int)
        right_point_idx = np.ceil(break_point).astype(int)
        new_point = np.mean([curve[left_point_idx, :], curve[right_point_idx, :]], axis=0)
        new_curve.append(new_point)
    new_curve.append(curve[-1,:])
    new_curve = np.array(new_curve)
    return new_curve


def expand_interpolate(curve, n_points):
    n_interval = len(curve) - 1
    n_new_points = n_points - len(curve)
    interval_point_insert = [0] * n_interval
    new_curve = [curve[0,:]]

    point_count = 0
    while True:
        for i in range(n_interval):
            point_count += 1
            interval_point_insert[i] += 1
            if point_count >= n_new_points:
                break
        if point_count >= n_new_points:
            break

    for i in range(n_interval):
        left_point = curve[i, :]
        right_point = curve[i+1, :]
        for k in range(interval_point_insert[i]):
            n_point_to_insert = interval_point_insert[i]
            new_point = left_point + (right_point - left_point) * k / n_point_to_insert
            new_curve.append(new_point)
        new_curve.append(right_point)
    new_curve = np.array(new_curve)
    return new_curve


def curve_interpolation(curve, n_points=1000):
    if len(curve) == n_points:
        return curve
    # if len(curve) > n_points:
    #     return reduce_interpolate(curve, n_points)
    # if len(curve) < n_points:
    #     return expand_interpolate(curve, n_points)
    n = curve.shape[0]
    x = curve[:, 0]
    y = curve[:, 1]
    z = curve[:, 2]
    x_points = np.linspace(np.min(x), np.max(x), n, endpoint=True)
    y_points = np.linspace(np.min(y), np.max(y), n, endpoint=True)
    z_points = np.linspace(np.min(z), np.max(z), n, endpoint=True)
    fx = interp1d(x_points, x, kind='linear')
    fy = interp1d(y_points, y, kind='linear')
    fz = interp1d(z_points, z, kind='linear')
    new_x = np.linspace(np.min(x), np.max(x), n_points, endpoint=True)
    new_y = np.linspace(np.min(y), np.max(y), n_points, endpoint=True)
    new_z = np.linspace(np.min(z), np.max(z), n_points, endpoint=True)
    x_interpolate = fx(new_x)
    y_interpolate = fy(new_y)
    z_interpolate = fz(new_z)

    curve_interpolate = np.vstack([x_interpolate, y_interpolate, z_interpolate])
    return curve_interpolate.T


def PCA_normalization(curve):
    curve = curve_interpolation(curve)
    pca = PCA(n_components=3)
    curve_pca = pca.fit_transform(curve)
    mass_center = np.mean(curve_pca, axis=0)
    curve_center = curve_pca - mass_center
    if np.abs(np.min(curve_center[:, 0])) > np.abs(np.max(curve_center[:, 0])):
        curve_center[:, 0] = -curve_center[:, 0]
    if np.abs(np.min(curve_center[:, 1])) > np.abs(np.max(curve_center[:, 1])):
        curve_center[:, 1] = -curve_center[:, 1]
    if np.abs(np.min(curve_center[:, 2])) > np.abs(np.max(curve_center[:, 2])):
        curve_center[:, 2] = -curve_center[:, 2]

    curve_center = curve_pca - curve_center[0, :]
    diagonal = curve_center[-1, :]
    rescale_factor = 1 / np.linalg.norm(diagonal)
    curve_rescale = curve_center * rescale_factor

    return curve_rescale


def fft_feature(curve, n_feature):
    fft_curve = fft(curve)
    features = fft_curve[0:n_feature]
    features = features.flatten()
    features_real = np.real(features).astype(float)
    features_imag = np.imag(features).astype(float)
    features = np.vstack([features_real, features_imag])
    features = features.T.flatten()
    return features, fft_curve


# def main():
#     curve, curve_normal = read_base_curve("train_data/Curve_in_base_frame.csv")
#     curve = curve[:, :]
#     curve_normalized = PCA_normalization(curve)
#     curve_fft_features, fft_all = fft_feature(curve_normalized, 5)
#     print(curve_fft_features)
#     print(fft_all)
#     print(fft_all.shape)
#
#
# if __name__ == '__main__':
#     main()
