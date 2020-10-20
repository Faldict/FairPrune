import numpy as np
from scipy.special import gamma, psi
from scipy import ndimage
from scipy.linalg import det
from numpy import pi
import pandas as pd
import responsibly

from sklearn.neighbors import NearestNeighbors

__all__ = [
    'entropy', 'mutual_information', 'entropy_gaussian',
    'mutual_information_2d', 'non_binary_equal_opportunity_difference',
    'add_noise', 'add_asymmetric_noise',
    'compute_mutual_information_between_X_and_A',
    'combine_X_A_as_dataframe'
]

EPS = np.finfo(float).eps


def add_noise(noise_rate, sensitive_features):
    """ Add noise to random binary sensitive attributes. 

    noise_rate: float, the percentage of samples that should be added with noise
    sensitive_features: array of int(0 or 1), sensitive attributes

    ret: array of int(0 or 1), sensitive attributes with noise
    """
    print('adding noise to sensitive features with noise_rate = ' +
          str(noise_rate))
    labels = np.unique(sensitive_features)
    assert len(labels) == 2
    m = len(sensitive_features)
    random_number_array = np.random.rand(m)
    for i in range(m):
        if noise_rate > random_number_array[i]:
            if sensitive_features[i] == labels[0]:
                sensitive_features[i] = labels[1]
            elif sensitive_features[i] == labels[1]:
                sensitive_features[i] = labels[0]
    return sensitive_features


def add_asymmetric_noise(noise_rate, sensitive_features):
    """ Add asymmetric noise to random binary sensitive attributes. 

    noise_rate: float, the percentage of samples that should be added with noise
                if they are 0
    sensitive_features: array of int(0 or 1), sensitive attributes

    ret: array of int(0 or 1), sensitive attributes with noise
    """
    print('adding asymmetric noise to sensitive features with noise_rate=' + str(noise_rate))
    labels = np.unique(sensitive_features)
    assert len(labels) == 2
    m = len(sensitive_features)
    random_number_array = np.random.rand(m)
    for i in range(m):
        if sensitive_features[i] == 0:
            continue
        if noise_rate > random_number_array[i]:
            if sensitive_features[i] == labels[0]:
                sensitive_features[i] = labels[1]
            elif sensitive_features[i] == labels[1]:
                sensitive_features[i] = labels[0]
    return sensitive_features


def nearest_distances(X, k=1):
    '''
    X = array(N,M)
    N = number of points
    M = number of dimensions
    returns the distance to the kth nearest neighbor for every point in X
    '''
    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(X)
    d, _ = knn.kneighbors(X)  # the first nearest neighbor is itself
    return d[:, -1]  # returns the distance to the kth nearest neighbor


def entropy_gaussian(C):
    '''
    Entropy of a gaussian variable with covariance matrix C
    '''
    if np.isscalar(C):  # C is the variance
        return .5 * (1 + np.log(2 * pi)) + .5 * np.log(C)
    else:
        n = C.shape[0]  # dimension
        return .5 * n * (1 + np.log(2 * pi)) + .5 * np.log(abs(det(C)))


def entropy(X, k=1):
    ''' Returns the entropy of the X.
    Parameters
    ===========
    X : array-like, shape (n_samples, n_features)
        The data the entropy of which is computed
    k : int, optional
        number of nearest neighbors for density estimation
    Notes
    ======
    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    See also: Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    and:
    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.
    '''

    # Distance to kth nearest neighbor
    r = nearest_distances(X, k)  # squared distances
    n, d = X.shape
    volume_unit_ball = (pi**(.5 * d)) / gamma(.5 * d + 1)
    '''
    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.
    return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
    '''
    return (d * np.mean(np.log(r + np.finfo(X.dtype).eps)) +
            np.log(volume_unit_ball) + psi(n) - psi(k))


def mutual_information(variables, k=1):
    '''
    Returns the mutual information between any number of variables.
    Each variable is a matrix X = array(n_samples, n_features)
    where
      n = number of samples
      dx,dy = number of dimensions
    Optionally, the following keyword argument can be specified:
      k = number of nearest neighbors for density estimation
    Example: mutual_information((X, Y)), mutual_information((X, Y, Z), k=5)
    '''
    if len(variables) < 2:
        raise AttributeError(
            "Mutual information must involve at least 2 variables")
    all_vars = np.hstack(variables)
    return (sum([entropy(X, k=k) for X in variables]) - entropy(all_vars, k=k))


def mutual_information_2d(x, y, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    bins = (64, 64)
    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant', output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2))) / np.sum(
            jh * np.log(jh))) - 1
    else:
        mi = (np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1)) -
              np.sum(s2 * np.log(s2)))

    return mi


def non_binary_equal_opportunity_difference(y_true, y_pred, sensitive_features):
    """ Report equal opportunity difference of non-binary groups
    """
    report = responsibly.fairness.metrics.report_binary(y_true, y_pred, sensitive_features)
    tpr = abs(report.loc['fnr'] - report.loc['fnr'].mean()).max()
    fpr = abs(report.loc['fpr'] - report.loc['fpr'].mean()).max()
    return max(tpr, fpr)


def equal_opportunity_difference(Y_true, Y_pred, sensitive_features):
    """
    Compute equal opportunity difference for a binary dataset.
    The difference is calculated by the abosolute value of 
    P(y_pred = 1| y_true = 1, sensitive_feature=1) 
    − P(y_pred = 1 | y_true = 1, sensitive_feature=0′)
    Parameters
    ----------
    y_true: 1D array
        ground truth labels
    y_pred : 1D array
        predicted labels
    sensitive features: 1D array
        sensitive features
    Returns
    -------
    equal_opportunity_difference: float
        the difference of equal opportunity
    """
    count1, count2, count3, count4 = 0, 0, 0, 0
    # Count1: count the case that y_true=1, y_pred=1, A=1
    # Count2: count the case that y_true=1, A=1
    # Count3: count the case that y_true=1, y_pred=1, A=0
    # Count4: count the case that y_true=1, A=0
    assert len(Y_true) == len(Y_pred) == len(sensitive_features)
    for i in range(len(Y_true)):
        if Y_true[i] != 1:
            continue
        if sensitive_features[i] == 1:
            count2 += 1
        elif sensitive_features[i] == 0:
            count4 += 1
        if Y_pred[i] != 1:
            continue
        if sensitive_features[i] == 1:
            count1 += 1
        elif sensitive_features[i] == 0:
            count3 += 1
    return abs(1. * count1 / count2 - 1. * count3 / count4)


def compute_mutual_information_between_X_and_A(X, Y, A, constraint, lambda_value):
    """ Compute mutual information between X and A

    The formula varies as constraint varies.
    """
    mis = []
    for col in X.columns:
        if constraint == 'EO':
            mi = mutual_information_2d(X[col].values[Y == 0],
                                       A[Y == 0]) + mutual_information_2d(
                                           X[col].values[Y == 1], A[Y == 1])
        else:
            mi = mutual_information_2d(
                X[col].values, A) - lambda_value * mutual_information_2d(
                    X[col].values, Y)
        mis.append((mi, col))
    mis = sorted(mis, reverse=False)
    mis1 = [l[1] for l in mis]
    return mis1


def combine_X_A_as_dataframe(X_train, A_train):
    """ Combine X and A together into a dataframe by matching their index.
    """
    A_train_df = pd.DataFrame(A_train, columns = ['sensitive_attribute'])
    print(A_train_df)
    result = pd.concat([X_train, A_train_df], axis=1, join='inner')
    result.index.names = ['sensitive_attribute']
    return result
