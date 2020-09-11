import numpy as np
import pandas as pd
import responsibly
from sklearn import preprocessing

__all__ = ['adult', 'compas', 'crime']

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
    X = pd.get_dummies(X).astype('float64')
    Y = compas_ds.df['is_recid'].fillna(0).values
    A = preprocessing.LabelEncoder().fit_transform(compas_ds.df['race'])
    return X, Y, A

def crime(sensitive_attribute=0):
    """ Community and Crime dataset
    sensitive attributes : racePctBlack (R), blackPerCapIncome (B), and pctNotSpeakEnglWell (P)
    """
    attrib = pd.read_csv('dataset/attributes.csv', delim_whitespace = True)
    data = pd.read_csv('dataset/communities.data', names = attrib['attributes'])
    data = data.drop(columns=['state','county',
                          'community','communityname',
                          'fold'], axis=1)
    data = data.replace('?', np.nan)
    from sklearn.impute import SimpleImputer 
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

    imputer = imputer.fit(data[['OtherPerCap']])
    data[['OtherPerCap']] = imputer.transform(data[['OtherPerCap']])
    data = data.dropna(axis=1)

    X = data.iloc[:, 0:100]
    Y = (data.iloc[:, 100].values > 0.333).astype(np.int64)

    X = pd.DataFrame(preprocessing.StandardScaler().fit_transform(X), columns=X.columns)
    sensitive_attributes = ['racepctblack', 'blackPerCap', 'PctNotSpeakEnglWell']
    A = data[sensitive_attributes[sensitive_attribute]]
    # A = preprocessing.StandardScaler().fit_transform(A)
    return X, Y, A
