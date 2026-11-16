"""Reviewers must see executable model-specific tuning, not a stale RF-only notebook."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_submission_notebook_has_inline_tuning_and_saved_outputs():
    path=ROOT/'notebooks/wolfcamp_midterm_submission.ipynb'
    assert path.read_bytes()==(ROOT/'notebooks/notebook.ipynb').read_bytes()
    notebook=json.loads(path.read_text(encoding='utf-8'))
    markdown='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='markdown')
    code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    for section in ['6.2. Ordinary linear regression','6.3. Tune Ridge regression',
                    '6.4. Tune Decision Tree regression','6.5. Tune Random Forest regression',
                    '6.6. Tune XGBoost regression','6.7. Select the final family']:
        assert section in markdown
    for estimator in ['LinearRegression(','Ridge(','DecisionTreeRegressor(',
                      'RandomForestRegressor(','XGBRegressor(']:
        assert estimator in code
    assert 'fold_pipeline.fit(X_fit,y_fit)' in code
    assert '## 6. Train and tune the Random Forest' not in markdown
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            assert cell['execution_count'] is not None
            assert cell['outputs']
            assert all(o['output_type']!='error' for o in cell['outputs'])
