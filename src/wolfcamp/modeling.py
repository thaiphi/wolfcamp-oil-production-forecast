"""Shared preprocessing and declared regression-family search spaces."""
import json
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from .features import MonthSixFeatures, ENGINEERED_NUMERIC


def preprocessor():
    numeric=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),
                      ('scaler',StandardScaler())])
    categorical=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                          ('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=10,sparse_output=False))])
    return ColumnTransformer([('numeric',numeric,NUMERIC_FEATURES+ENGINEERED_NUMERIC),
                              ('categorical',categorical,CATEGORICAL_FEATURES)],
                              remainder='drop',sparse_threshold=0)


def make_pipeline(params=None, family='random_forest'):
    settings={'n_estimators':160,'max_features':.75,'max_samples':.7,
              'random_state':42,'n_jobs':2,'max_depth':8,'min_samples_leaf':2}
    if family=='random_forest':
        settings.update(params or {})
        estimator=RandomForestRegressor(**settings)
    elif family=='linear_regression':
        estimator=LinearRegression(**(params or {}))
    elif family=='ridge':
        estimator=Ridge(**(params or {}))
    elif family=='decision_tree':
        estimator=DecisionTreeRegressor(random_state=42,**(params or {}))
    elif family=='xgboost':
        estimator=XGBRegressor(objective='reg:squarederror',tree_method='hist',random_state=42,n_jobs=2,**(params or {}))
    else:
        raise ValueError('Unknown model family: '+family)
    return Pipeline([('domain_features',MonthSixFeatures()),('preprocess',preprocessor()),
                     ('model',estimator)])


def candidate_specs():
    """One OLS benchmark and twelve trials per tunable model family (49 total).

    RF preserves the six already validated settings and adds six deterministic
    random settings. Search spaces are declared before the new evaluation.
    """
    yield {'family':'linear_regression','params':{}}
    for alpha in [.01,.03,.1,.3,1.,3.,10.,30.,100.,300.,1000.,3000.]:
        yield {'family':'ridge','params':{'alpha':alpha}}
    for depth in [3,5,8,12]:
        for leaf in [2,5,15]:
            yield {'family':'decision_tree','params':{'max_depth':depth,'min_samples_leaf':leaf}}
    seen=set()
    for depth in [5,8,12]:
        for leaf in [2,5]:
            params={'n_estimators':160,'max_depth':depth,'min_samples_leaf':leaf,'max_features':.75,'max_samples':.7}
            seen.add(json.dumps(params,sort_keys=True))
            yield {'family':'random_forest','params':params}
    space={'n_estimators':[120,160,240],'max_depth':[5,8,12,None],
           'min_samples_leaf':[1,2,5,10],'max_features':[.5,.75,1.],'max_samples':[.7,.9,None]}
    for params in ParameterSampler(space,n_iter=100,random_state=42):
        key=json.dumps(params,sort_keys=True)
        if key not in seen:
            seen.add(key)
            yield {'family':'random_forest','params':params}
        if len(seen)==12: break
    space={'n_estimators':[100,200,350],'learning_rate':[.025,.05,.1],'max_depth':[2,3,4],
           'min_child_weight':[1,3,8],'subsample':[.65,.8,1.],'colsample_bytree':[.65,.8,1.],
           'reg_lambda':[1,10,40,100],'reg_alpha':[0,1,10]}
    for params in ParameterSampler(space,n_iter=12,random_state=42):
        yield {'family':'xgboost','params':params}


def score(y_true,predicted):
    predicted=np.maximum(0,np.asarray(predicted,dtype=float))
    return {'mae_bbl':round(float(mean_absolute_error(y_true,predicted)),2),
            'rmse_bbl':round(float(np.sqrt(mean_squared_error(y_true,predicted))),2),
            'r2':round(float(r2_score(y_true,predicted)),4)}


class RecentRateBaseline(RegressorMixin,BaseEstimator):
    """Train-fitted scale on twice the latest three-month oil production."""
    @staticmethod
    def _flat_forecast(X):
        recent=np.maximum(0,np.asarray(X['First6MonthOil_BBL'],dtype=float)-np.asarray(X['First3MonthOil_BBL'],dtype=float))
        return 2*np.nan_to_num(recent,nan=0,posinf=0,neginf=0)

    def fit(self,X,y):
        x=self._flat_forecast(X)
        denominator=float(x@x)
        self.scale_=max(0,float(x@np.asarray(y,dtype=float))/denominator) if denominator>0 else 0
        return self

    def predict(self,X):
        return self.scale_*self._flat_forecast(X)
