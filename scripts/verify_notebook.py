"""Execute the submission notebook in memory, preserving its saved bytes."""
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    for var,name in [('IPYTHONDIR','ipython'),('JUPYTER_RUNTIME_DIR','jupyter'),('JUPYTER_CONFIG_DIR','jupyter-config'),('MPLCONFIGDIR','matplotlib')]:
        path=ROOT/'reports/verification/runtime'/name; path.mkdir(parents=True,exist_ok=True); os.environ[var]=str(path)
    notebook=nbformat.read(ROOT/'notebooks/wolfcamp_midterm_submission.ipynb',as_version=4)
    NotebookClient(notebook,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    print('PASS:',sum(c.cell_type=='code' for c in notebook.cells),'code cells executed; zero errors')
