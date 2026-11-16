"""Create a portable checksum manifest for the submission's validated artifacts."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    names=['data/raw/wolfcamp_wells.csv','models/model.joblib','notebooks/notebook.ipynb',
           'notebooks/wolfcamp_midterm_submission.ipynb',
           'reports/metrics.json','reports/tuning.json','reports/tuning.csv','reports/model_selection.csv',
           'reports/selection.json','reports/MODEL_SELECTION_PROTOCOL.md','reports/platform_reproducibility.json']
    entries=[{'path':name,'sha256':hashlib.sha256((ROOT/name).read_bytes()).hexdigest()} for name in names]
    (ROOT/'artifact_manifest.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
    print('Frozen',len(entries),'artifacts')
