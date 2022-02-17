import numpy as np
from scipy.fft import fft

from sklearn.decomposition import PCA


def PCA_normalization(curve):
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
    features = np.abs(features)
    return features
