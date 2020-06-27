import numpy as np
import pandas as pd
import responsibly
from sklearn import preprocessing

__all__ = ['adult', 'compas']

def adult():
    """ Adult dataset
    return:
        X - DataFrame
        Y - Numpy Array
        A - Numpy Array
    """
    from responsibly.dataset import AdultDataset

    adult_ds = AdultDataset()
    features = ['age', 'workclass', 'education', 'education-num', 'marital_status', 'occupation', 'relationship', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
    X = pd.get_dummies(adult_ds.df[features])
    X = pd.DataFrame(preprocessing.StandardScaler().fit_transform(X), columns=X.columns)
    Y = preprocessing.LabelEncoder().fit_transform(adult_ds.df[adult_ds.target])
    A = preprocessing.LabelEncoder().fit_transform(adult_ds.df['sex'])
    return X, Y, A

def compas():
    """ COMPAS dataset
    """
    from responsibly.dataset import COMPASDataset
    compas_ds = COMPASDataset()
    X = compas_ds.df[['sex', 'age', 'c_charge_degree', 'age_cat', 'score_text', 'priors_count', 'days_b_screening_arrest', 'decile_score', 'length_of_stay']]
    X.loc[:, 'length_of_stay'] = X['length_of_stay'].dt.days
    X = X.fillna(0)
    X = pd.get_dummies(X)
    Y = compas_ds.df['is_recid'].values
    A = preprocessing.LabelEncoder().fit_transform(compas_ds.df['race'])
    return X, Y, A
