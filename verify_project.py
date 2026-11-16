"""Verify portable artifact hashes, reported metrics and optional retraining parity."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath,PureWindowsPath
import joblib
import numpy as np
from src.wolfcamp.data import get_xy,split_by_year
from src.wolfcamp.study import study_data
from src.wolfcamp.modeling import score

ROOT=Path(__file__).resolve().parent


def verify(rerun=None,allow_platform_differences=False):
    entries=json.loads((ROOT/'artifact_manifest.json').read_text())
    for entry in entries:
        name=entry['path']
        if PurePosixPath(name).is_absolute() or PureWindowsPath(name).drive or '..' in PurePosixPath(name).parts or '\\' in name:
            raise ValueError('Nonportable manifest path: '+name)
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Artifact checksum mismatch: '+name)
    if not entries: raise ValueError('Empty artifact manifest')
    metrics=json.loads((ROOT/'reports/metrics.json').read_text())
    train,valid,test=split_by_year(study_data())
    for key,value in [('train',len(train)),('validation',len(valid)),('historical_holdout',len(test))]:
        if metrics['counts'][key]!=value: raise ValueError('Cohort count differs: '+key)
    x,y=get_xy(test)
    model=joblib.load(ROOT/'models/model.joblib')
    actual=score(y,model.predict(x))
    if actual!=metrics['historical_holdout_metrics']: raise ValueError('Packaged model metrics differ')
    result={'status':'PASS','artifacts_verified':len(entries),'historical_holdout_metrics':actual}
    if rerun:
        rerun=Path(rerun)
        reported=json.loads((rerun/'metrics.json').read_text())
        for key in ['model','selected_family','best_params','counts','validation_metrics','historical_holdout_metrics']:
            if metrics[key]!=reported[key]: raise ValueError('Retraining differs: '+key)
        original_trials=json.loads((ROOT/'reports/tuning.json').read_text())
        new_trials=json.loads((rerun/'tuning.json').read_text())
        if len(original_trials)!=49 or len(new_trials)!=49: raise ValueError('Expected 49 tuning trials')
        differences=[]
        for a,b in zip(original_trials,new_trials,strict=True):
            if a['params']!=b['params'] or a['family']!=b['family']: raise ValueError('Trial specification differs')
            aa=np.asarray([f['rmse_bbl'] for f in a['folds']])
            bb=np.asarray([f['rmse_bbl'] for f in b['folds']])
            if np.max(np.abs(aa-bb))>.01:
                if a['family']==metrics['selected_family'] or not allow_platform_differences:
                    raise ValueError('Trial fold scores differ: '+a['family']+' '+json.dumps(a['params']))
                differences.append({'family':a['family'],'params':a['params'],
                    'packaged_fold_rmse_bbl':aa.tolist(),'rerun_fold_rmse_bbl':bb.tolist(),
                    'max_absolute_difference_bbl':float(np.max(np.abs(aa-bb)))})
        np.testing.assert_allclose(model.predict(x),joblib.load(rerun/'model.joblib').predict(x),rtol=0,atol=1e-8)
        result['retraining_matches']=True
        result['tuning_trials_verified']=49
        result['all_trial_scores_match_within_0_01_bbl']=not differences
        result['nonselected_platform_differences']=differences
        if differences: result['status']='PASS_WITH_PLATFORM_DIFFERENCES'
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rerun',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--allow-platform-differences',action='store_true',help='Report nonwinning-family numerical differences; selected-family and final-artifact parity remain strict')
    args=parser.parse_args()
    text=json.dumps(verify(args.rerun,args.allow_platform_differences),indent=2)
    if args.output: args.output.write_text(text,encoding='utf-8')
    print(text)
