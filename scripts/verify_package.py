"""Validate upload ZIP checksums and parity with a tested clean extraction."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]


def verify(archive,tested=None):
    prefix='wolfcamp-midterm-submission/'
    with ZipFile(archive) as bundle:
        entries=json.loads(bundle.read(prefix+'submission_files.json'))
        runtime=[]
        for entry in entries:
            name=entry['path']
            if hashlib.sha256(bundle.read(prefix+name)).hexdigest()!=entry['sha256']:
                raise ValueError('ZIP checksum mismatch: '+name)
            if set(Path(name).parts)&{'.venv','.verify-env','__pycache__','.pytest_cache','clean-copy'}:
                raise ValueError('Excluded workspace content included: '+name)
            if name.startswith(('src/','tests/')) or name in {'train.py','predict.py','verify_project.py',
                'requirements.txt','requirements-dev.txt','Dockerfile','models/model.joblib','notebooks/notebook.ipynb',
                'notebooks/wolfcamp_midterm_submission.ipynb'}:
                runtime.append(name)
        if bundle.read(prefix+'notebooks/notebook.ipynb')!=bundle.read(prefix+'notebooks/wolfcamp_midterm_submission.ipynb'):
            raise ValueError('Submission notebook aliases differ')
        if tested:
            for name in runtime:
                if bundle.read(prefix+name)!=(tested/name).read_bytes():
                    raise ValueError('Tested runtime differs: '+name)
    return {'status':'PASS','files_verified':len(entries),'tested_runtime_files_compared':len(runtime) if tested else 0,
            'environments_and_caches_excluded':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--tested',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    text=json.dumps(verify(args.archive,args.tested),indent=2)
    if args.output: args.output.write_text(text,encoding='utf-8')
    print(text)
