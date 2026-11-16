"""Package the standalone project with an explicit checksum list."""
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parent


def files_to_package():
    names=['README.md','SUBMISSION.md','VERIFICATION.md','train.py','predict.py','verify_project.py',
           'prepare_submission.py','requirements.txt','requirements-dev.txt','Dockerfile','render.yaml','docs/render-deployment.md','pytest.ini',
           '.gitignore','.dockerignore','.gitattributes','artifact_manifest.json','provenance.json',
           'data/raw/wolfcamp_wells.csv','data/raw/README.md','models/model.joblib','notebooks/notebook.ipynb',
           'notebooks/wolfcamp_midterm_submission.ipynb',
           'reports/metrics.json','reports/tuning.json','reports/tuning.csv','reports/model_selection.csv',
           'reports/selection.json','reports/MODEL_SELECTION_PROTOCOL.md','scripts/Dockerfile.tests',
           'docs/docker-prediction.png','docs/render-prediction.png','docs/workflow.svg',
           'docs/workflow.html','docs/workflow.archify.json','docs/archify-LICENSE.txt',
           'docs/archify-LICENSE-MIT.txt','reports/platform_reproducibility.json']
    files=[ROOT/name for name in names]
    for directory in ['src','tests','examples','scripts']:
        files += [p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ['.py','.json'] and '__pycache__' not in p.parts]
    files += [p for p in (ROOT/'reports/verification').glob('*') if p.is_file() and p.suffix in ['.json','.log']]
    for path in files:
        if not path.is_file(): raise FileNotFoundError(path)
    return sorted(set(files))


def main():
    files=files_to_package()
    destination=ROOT/'dist/wolfcamp-midterm-submission'
    allowed={p.relative_to(ROOT).as_posix() for p in files}|{'submission_files.json'}
    allowed_dirs={p.as_posix() for name in allowed for p in Path(name).parents}
    if destination.exists():
        for path in destination.rglob('*'):
            name=path.relative_to(destination).as_posix()
            if path.is_symlink() or (path.is_file() and name not in allowed) or (path.is_dir() and name not in allowed_dirs):
                raise ValueError('Unlisted upload content; move it elsewhere: '+name)
    entries=[]
    for path in files:
        name=path.relative_to(ROOT)
        target=destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        entries.append({'path':name.as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (destination/'submission_files.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
    archive=ROOT/'dist/wolfcamp-midterm-submission-github.zip'
    with ZipFile(archive,'w',compression=ZIP_DEFLATED) as bundle:
        for name in sorted(allowed): bundle.write(destination/name,'wolfcamp-midterm-submission/'+name)
    checksum=hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(checksum+'  '+archive.name+'\n',encoding='utf-8')
    print(json.dumps({'folder':str(destination),'archive':str(archive),'files':len(allowed),'sha256':checksum},indent=2))


if __name__=='__main__': main()
