"""Purged chronological tuning, validation, final refit and evaluation."""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from .data import DEFAULT_DATA,get_xy,split_by_year
from .study import study_data,purged_fit
from .modeling import make_pipeline,score,RecentRateBaseline,candidate_specs


def training_folds(train):
    for year in (2016,2017,2018):
        scoring=train.loc[train.FirstProdYear==year]
        yield year,purged_fit(train.loc[train.FirstProdYear<year],scoring),scoring


def select_best(rows):
    """Rank every family by training CV; no preferred estimator family."""
    return min(rows,key=lambda r:(r['cv_rmse_bbl'],r['family'],json.dumps(r['params'],sort_keys=True)))


def tune(train,verbose=True):
    rows=[]
    for spec in candidate_specs():
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always')
            folds=[]
            for year,fit,scoring in training_folds(train):
                model=make_pipeline(spec['params'],family=spec['family']).fit(*get_xy(fit))
                x,y=get_xy(scoring)
                predicted=np.maximum(0,model.predict(x))
                folds.append({'year':year,'fit_rows':len(fit),'scoring_rows':len(scoring),
                    'purged_rows':int((train.FirstProdYear<year).sum())-len(fit),
                    'rmse_bbl':float(root_mean_squared_error(y,predicted)),
                    'fit_rmse_bbl':score(get_xy(fit)[1],model.predict(get_xy(fit)[0]))['rmse_bbl']})
            row=spec|{'cv_rmse_bbl':float(np.mean([r['rmse_bbl'] for r in folds])),'folds':folds,
                      'warnings':sorted({str(m.message) for m in messages})}
            rows.append(row)
            if verbose:
                print(spec['family'],spec['params'],'CV RMSE',round(row['cv_rmse_bbl'],2),flush=True)
    best=select_best(rows)
    return best,rows


def run(output,data=DEFAULT_DATA):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    frame=study_data(data)
    train,valid,test=split_by_year(frame)
    best,trials=tune(train)
    selection={k:best[k] for k in ['family','params','cv_rmse_bbl']}
    (output/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    fit_valid=purged_fit(train,valid)
    xv,yv=get_xy(valid)
    comparison=[]
    for family in dict.fromkeys(r['family'] for r in trials):
        family_rows=[r for r in trials if r['family']==family]
        family_best=min(family_rows,key=lambda r:(r['cv_rmse_bbl'],json.dumps(r['params'],sort_keys=True)))
        validation_model=make_pipeline(family_best['params'],family=family).fit(*get_xy(fit_valid))
        comparison.append({'family':family,'trials':len(family_rows),'best_params':family_best['params'],
                           'cv_rmse_bbl':family_best['cv_rmse_bbl'],
                           'validation_metrics':score(yv,validation_model.predict(xv))})
    validation=next(r['validation_metrics'] for r in comparison if r['family']==best['family'])
    final_fit=purged_fit(pd.concat([train,valid]),test)
    model=make_pipeline(best['params'],family=best['family']).fit(*get_xy(final_fit))
    model_path=output/'model.joblib'
    joblib.dump(model,model_path)
    saved=joblib.load(model_path)
    xt,yt=get_xy(test)
    np.testing.assert_allclose(model.predict(xt),saved.predict(xt),rtol=0,atol=1e-8)
    recent=RecentRateBaseline().fit(*get_xy(final_fit))
    results={'model':type(model.named_steps['model']).__name__,'selected_family':best['family'],'best_params':best['params'],
             'model_selection':comparison,'trial_count':len(trials),'fold_fit_count':sum(len(r['folds']) for r in trials),
             'cv_rmse_bbl':best['cv_rmse_bbl'],'validation_metrics':validation,
             'historical_holdout_metrics':score(yt,saved.predict(xt)),
             'historical_holdout_recent_rate_baseline':score(yt,recent.predict(xt)),
             'historical_holdout_median_baseline':score(yt,np.repeat(np.median(get_xy(final_fit)[1]),len(test))),
             'counts':{'train':len(train),'validation':len(valid),'historical_holdout':len(test),
                       'validation_fit_after_purge':len(fit_valid),'final_fit_after_purge':len(final_fit)},
             'completion_masked_by_cohort':{name:int(f.completion_masked.sum()) for name,f in
                                          [('train',train),('validation',valid),('historical_holdout',test)]},
             'limitations':['Previously inspected historical holdout; not independent prospective evidence.',
                            'Calendar maturity approximates reporting availability; vendor revisions unknown.',
                            'Callers must enforce the month-six input contract and study domain.']}
    (output/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    (output/'tuning.json').write_text(json.dumps(trials,indent=2),encoding='utf-8')
    pd.DataFrame([{'family':r['family'],'params':json.dumps(r['params'],sort_keys=True),
                   'cv_rmse_bbl':r['cv_rmse_bbl']} for r in trials]).to_csv(output/'tuning.csv',index=False)
    pd.DataFrame([{'family':r['family'],'trials':r['trials'],'cv_rmse_bbl':r['cv_rmse_bbl']}|
                   {'validation_'+k:v for k,v in r['validation_metrics'].items()} for r in comparison]).to_csv(output/'model_selection.csv',index=False)
    return results
