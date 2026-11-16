"""Study eligibility, month-six availability masks and calendar-label purging."""
import numpy as np
import pandas as pd
from .data import DEFAULT_DATA, load_data

COMPLETION_FIELDS = ['Proppant_LBS', 'ProppantIntensity_LBSPerFT', 'TotalWaterPumped_GAL',
                     'WaterIntensity_GALPerFT', 'FracStages', 'CompletionTime_DAYS', 'SoakTime_DAYS',
                     'ENVFluidType', 'ENVFracJobType', 'ENVProppantType']


def study_data(path=DEFAULT_DATA):
    frame = load_data(path)
    extra = pd.read_csv(path, usecols=['CompletionID', 'CompletionDate', 'TotalProducingMonths'], low_memory=False)
    if extra.CompletionID.duplicated().any():
        raise ValueError('CompletionID must uniquely identify source metadata')
    frame = frame.merge(extra, on='CompletionID', how='left', validate='one_to_one')
    eligible = (frame.TVD_FT.between(7000,15000) & frame.LateralLength_FT.between(4000,16000)
                & (pd.to_numeric(frame.TotalProducingMonths, errors='coerce') >= 12))
    frame = frame.loc[eligible].copy()
    completion = pd.to_datetime(frame.CompletionDate, errors='coerce')
    frame['completion_masked'] = completion.isna() | (completion > frame.FirstProdDate + pd.DateOffset(months=6))
    for col in COMPLETION_FIELDS:
        frame.loc[frame.completion_masked, col] = np.nan
    return frame.sort_values(['FirstProdDate', 'WellID']).reset_index(drop=True)


def purged_fit(fit, scoring):
    cutoff = scoring.FirstProdDate.min() + pd.DateOffset(months=6)
    return fit.loc[fit.FirstProdDate + pd.DateOffset(months=12) <= cutoff].copy()
