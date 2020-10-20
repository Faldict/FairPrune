import numpy as np
import pandas as pd
import responsibly
from sklearn import preprocessing

__all__ = ['adult_sex', 'adult_race', 'adult_sex_race', 'compas', 'celeba', 'celeba_young_male']


def np2darray_to_df(X):
    """ Convert 2d numpy array to pandas dataframe.
    """
    df_dict = {}
    for i in range(len(X[0])):
        df_dict['Column' + str(i)] = X[:, i]
    return pd.DataFrame(df_dict)


def unique_label(feature1, feature2):
    """ Convert two sensitive attributes into one sensitive attribtues. 

    For example, if feature1 and feature2 are two sensitive attribtues. 
    And there is 
    >>> feature1 = [0, 0, 1, 1]
    >>> feature2 = [0, 1, 0, 1]
    >>> unique_label(feature1, feature2)
    array([0, 1, 2, 3])
    """
    index1 = np.unique(feature1) 
    index2 = np.unique(feature2)
    res = []
    assert len(feature1) == len(feature2)
    for i in range(len(feature1)):
        res.append(int(feature1[i] * len(index2) + feature2[i]))
    return np.array(res)


def celeba():
    """ Load CelebA dataset. 

    Import CelebA dataset and use gender as sensitive attributes. 
    """
    X_train = np2darray_to_df(np.load('celeba/celeba.train.npy'))
    X_test = np2darray_to_df(np.load('celeba/celeba.test.npy'))
    target_train = np.load('celeba/celeba_label.train.npy')
    target_test = np.load('celeba/celeba_label.test.npy')
    Y_train = target_train[:, 31]
    Y_test = target_test[:, 31]
    A_train = target_train[:, 20]
    A_test = target_test[:, 20]
    return X_train, X_test, Y_train, Y_test, A_train, A_test


def celeba_young_male():
    """ Load CelebA dataset. 

    Import CelebA dataset and use gender and young as sensitive attributes. 
    """
    X_train = np2darray_to_df(np.load('celeba/celeba.train.npy'))
    X_test = np2darray_to_df(np.load('celeba/celeba.test.npy'))
    target_train = np.load('celeba/celeba_label.train.npy')
    target_test = np.load('celeba/celeba_label.test.npy')
    Y_train = target_train[:, 31]
    Y_test = target_test[:, 31]
    A_train = unique_label(target_train[:, 20], target_train[:, -1])
    A_test = unique_label(target_test[:, 20], target_test[:, -1])
    return X_train, X_test, Y_train, Y_test, A_train, A_test


def adult_sex_race():
    """ Adult dataset - sex and race as sensitive attribtues
    return:
        X - DataFrame
        Y - Numpy Array
        A - Numpy Array
    """
    print('Fetching Adult data with sensitive_attribute=sex, race ...')

    from responsibly.dataset import AdultDataset

    adult_ds = AdultDataset()
    features = ['age', 'workclass', 'education', 'education-num', 'marital_status', 'occupation', 'relationship', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
    X = pd.get_dummies(adult_ds.df[features])
    X = pd.DataFrame(preprocessing.StandardScaler().fit_transform(X), columns=X.columns)
    Y = preprocessing.LabelEncoder().fit_transform(adult_ds.df[adult_ds.target])
    A = unique_label(preprocessing.LabelEncoder().fit_transform(adult_ds.df['sex']), 
                    preprocessing.LabelEncoder().fit_transform(adult_ds.df['race']))

    return X, Y, A


def adult_sex():
    """ Adult dataset - sex as sensitive attribtues
    return:
        X - DataFrame
        Y - Numpy Array
        A - Numpy Array
    """
    print('Fetching Adult data with sensitive_attribute=sex ...')

    from responsibly.dataset import AdultDataset

    adult_ds = AdultDataset()
    features = ['age', 'workclass', 'education', 'education-num', 'marital_status', 'occupation', 'relationship', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
    X = pd.get_dummies(adult_ds.df[features])
    X = pd.DataFrame(preprocessing.StandardScaler().fit_transform(X), columns=X.columns)
    Y = preprocessing.LabelEncoder().fit_transform(adult_ds.df[adult_ds.target])
    A = preprocessing.LabelEncoder().fit_transform(adult_ds.df['sex'])

    return X, Y, A


def adult_race():
    """ Adult dataset - race as sensitive attributes
    return:
        X - DataFrame
        Y - Numpy Array
        A - Numpy Array
    """
    print('Fetching Adult data with sensitive_attribute=race ...')

    from responsibly.dataset import AdultDataset

    adult_ds = AdultDataset()
    features = ['age', 'workclass', 'education', 'education-num', 'marital_status', 'occupation', 'relationship', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
    X = pd.get_dummies(adult_ds.df[features])
    X = pd.DataFrame(preprocessing.StandardScaler().fit_transform(X), columns=X.columns)
    Y = preprocessing.LabelEncoder().fit_transform(adult_ds.df[adult_ds.target])
    A = preprocessing.LabelEncoder().fit_transform(adult_ds.df['race'])

    return X, Y, A


def compas():
    """ COMPAS dataset - race as sensitive attributes
    """
    print('Fetching COMPAS data with sensitive_attribute=race ...')

    from responsibly.dataset import COMPASDataset

    compas_ds = COMPASDataset()
    X = compas_ds.df[['sex', 'age', 'c_charge_degree', 'age_cat', 'score_text', 'priors_count', 'days_b_screening_arrest', 'decile_score', 'length_of_stay']]
    X.loc[:, 'length_of_stay'] = X['length_of_stay'].dt.days
    X = X.fillna(0)
    X = pd.get_dummies(X).astype('float64')
    Y = compas_ds.df['is_recid'].fillna(0).values
    A = preprocessing.LabelEncoder().fit_transform(compas_ds.df['race'])
    return X, Y, A
