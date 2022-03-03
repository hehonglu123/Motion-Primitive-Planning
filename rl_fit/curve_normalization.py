import numpy as np
from scipy.fft import fft

from sklearn.decomposition import PCA
from trajectory_utils import read_base_curve

from plot_functions import plot_curve


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


def normalize_points(curve, n_points=1000):
    if len(curve) == n_points:
        return curve
    if len(curve) > n_points:
        return reduce_interpolate(curve, n_points)
    if len(curve) < n_points:
        return expand_interpolate(curve, n_points)


def PCA_normalization(curve):
    curve = normalize_points(curve)
    pca = PCA(n_components=3)
    curve_pca = pca.fit_transform(curve)
    curve_center = curve_pca - curve_pca[0, :]
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


def main():
    curve, curve_normal = read_base_curve("data/Curve_in_base_frame.csv")
    curve = curve[:, :]
    curve_normalized = PCA_normalization(curve)
    curve_fft_features, fft_all = fft_feature(curve_normalized, 5)
    print(curve_fft_features)
    print(fft_all)
    print(fft_all.shape)


if __name__ == '__main__':
    main()
