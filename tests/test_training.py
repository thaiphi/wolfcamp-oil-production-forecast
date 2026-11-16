import numpy as np
import pandas as pd
import joblib
from src.wolfcamp.data import FEATURES, get_xy, split_by_year, TARGET
from src.wolfcamp.study import study_data, purged_fit, COMPLETION_FIELDS
from src.wolfcamp.modeling import make_pipeline, score
from src.wolfcamp.training import training_folds,select_best
from src.wolfcamp.modeling import candidate_specs


def test_cohort_and_purge():
    frame=study_data()
    train,valid,test=split_by_year(frame)
    assert [len(train),len(valid),len(test)] == [1249,428,264]
    assert frame.WellID.is_unique
    assert (frame[TARGET]>=0).all()
    assert frame.loc[frame.completion_masked,COMPLETION_FIELDS].isna().all().all()
    for year,fit,scoring in training_folds(train):
        assert fit.FirstProdYear.max()<year
        assert set(fit.WellID).isdisjoint(scoring.WellID)
        assert (fit.FirstProdDate+pd.DateOffset(months=12) <= scoring.FirstProdDate.min()+pd.DateOffset(months=6)).all()


def test_exact_purge_boundary():
    fit=pd.DataFrame({'FirstProdDate':pd.to_datetime(['2017-07-01','2017-07-02'])})
    scoring=pd.DataFrame({'FirstProdDate':pd.to_datetime(['2018-01-01'])})
    assert len(purged_fit(fit,scoring)) == 1


def test_training_only_preprocessing_and_unknown_inputs():
    frame=study_data()
    x,y=get_xy(frame.iloc[:180])
    model=make_pipeline({'max_depth':5,'min_samples_leaf':5}).fit(x,y)
    imputer=model.named_steps['preprocess'].named_transformers_['numeric'].named_steps['imputer']
    original=imputer.statistics_.copy()
    unseen=x.iloc[:2].copy()
    unseen['County']='UNSEEN'
    unseen['Proppant_LBS']=np.nan
    unseen['First3MonthOil_BBL']=np.nan
    assert len(model.predict(unseen))==2
    assert np.isfinite(model.predict(unseen)).all()
    np.testing.assert_array_equal(imputer.statistics_,original)
    assert all('12Month' not in name and '9Month' not in name and '36Month' not in name for name in FEATURES)


def test_saved_model_reproduces_validated_score():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    _,_,test=split_by_year(study_data())
    x,y=get_xy(test)
    model=joblib.load(root/'models/model.joblib')
    import json
    expected=json.loads((root/'reports/metrics.json').read_text())
    assert score(y,model.predict(x)) == expected['historical_holdout_metrics']
    assert type(model.named_steps['model']).__name__ == expected['model']


def test_search_includes_all_families_and_preserves_validated_rf_candidate():
    from collections import Counter
    specs=list(candidate_specs())
    assert Counter(s['family'] for s in specs) == {'linear_regression':1,'ridge':12,'decision_tree':12,'random_forest':12,'xgboost':12}
    assert any(s['family']=='random_forest' and s['params'].get('max_depth')==8 and s['params'].get('min_samples_leaf')==2
               and s['params'].get('n_estimators')==160 for s in specs)


def test_every_family_has_same_preprocessing():
    for family in ['linear_regression','ridge','decision_tree','random_forest','xgboost']:
        model=make_pipeline(family=family)
        assert list(model.named_steps)==['domain_features','preprocess','model']
        assert model.named_steps['preprocess'].get_params()['numeric__imputer__strategy']=='median'


def test_selection_does_not_prefer_random_forest():
    rows=[{'family':'random_forest','params':{},'cv_rmse_bbl':30000},
          {'family':'ridge','params':{},'cv_rmse_bbl':29000},
          {'family':'xgboost','params':{},'cv_rmse_bbl':25000}]
    assert select_best(rows)['family']=='xgboost'
    rows[1]['cv_rmse_bbl']=24000
    assert select_best(rows)['family']=='ridge'
