"""Check HTTP predictions against the packaged pipeline on representative inputs."""
import argparse
import json
from pathlib import Path
import sys
from urllib.request import Request,urlopen
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.wolfcamp.data import FEATURES,get_xy,split_by_year
from src.wolfcamp.study import study_data


def verify(url):
    _,_,test=split_by_year(study_data(ROOT/'data/raw/wolfcamp_wells.csv'))
    x,_=get_xy(test)
    sample={k:None if pd.isna(v) else v.item() if hasattr(v,'item') else v for k,v in x.iloc[0].items()}
    cases=[sample,{'First6MonthOil_BBL':1000},sample|{'County':'UNSEEN COUNTY','FirstProdYear':2099},
           sample|{'First3MonthOil_BBL':0,'First3MonthWater_BBL':0},sample|{'ENVFluidType':None,'FracStages':None}]
    model=joblib.load(ROOT/'models/model.joblib')
    results=[]
    for i,values in enumerate(cases):
        row=pd.DataFrame([{k:np.nan if values.get(k) is None else values[k] for k in FEATURES}])
        expected=round(max(0,float(model.predict(row)[0])),2)
        request=Request(url.rstrip('/')+'/predict',data=json.dumps({'features':values}).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=30) as reply: actual=json.load(reply)['predicted_months_7_to_12_oil_bbl']
        if abs(expected-actual)>.01: raise AssertionError(f'Case {i}: artifact {expected}, API {actual}')
        results.append({'case':i,'artifact':expected,'api':actual,'match':True})
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8000')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    text=json.dumps(verify(args.url),indent=2)
    if args.output: args.output.write_text(text,encoding='utf-8')
    print(text)
