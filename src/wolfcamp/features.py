"""Deterministic features from supplied month-six snapshots; no target inputs."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ENGINEERED_NUMERIC = [
    "oil_months_4_to_6_bbl", "gas_months_4_to_6_mcf", "water_months_4_to_6_bbl",
    "oil_recent_early_ratio", "oil_6_per_lateral_ft", "oil_3_per_lateral_ft",
    "water_fraction_6", "gas_oil_ratio_6_mcf_per_bbl",
    "frac_stages_per_1000ft", "log_proppant_per_stage_lbs", "log_fluid_per_stage_gal",
]


def _divide(n, d):
    n, d = pd.to_numeric(n, errors="coerce"), pd.to_numeric(d, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        result = n / d.where(d > 0)
    return result.replace([np.inf, -np.inf], np.nan)


class MonthSixFeatures(BaseEstimator, TransformerMixin):
    """Deterministic feature construction. Fit does not learn from held-out data.

    Input contains only the 25 approved month-six features. All output columns
    must be computable from this same input, so inference and CV agree.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        oil3, oil6 = X["First3MonthOil_BBL"], X["First6MonthOil_BBL"]
        gas3, gas6 = X["First3MonthGas_MCF"], X["First6MonthGas_MCF"]
        water3, water6 = X["First3MonthWater_BBL"], X["First6MonthWater_BBL"]
        length, stages = X["LateralLength_FT"], X["FracStages"]
        X["oil_months_4_to_6_bbl"] = (oil6 - oil3).clip(lower=0)
        X["gas_months_4_to_6_mcf"] = (gas6 - gas3).clip(lower=0)
        X["water_months_4_to_6_bbl"] = (water6 - water3).clip(lower=0)
        X["oil_recent_early_ratio"] = _divide(X["oil_months_4_to_6_bbl"], oil3).clip(lower=0, upper=10)
        X["oil_6_per_lateral_ft"] = _divide(oil6, length)
        X["oil_3_per_lateral_ft"] = _divide(oil3, length)
        X["water_fraction_6"] = _divide(water6, oil6 + water6)
        X["gas_oil_ratio_6_mcf_per_bbl"] = _divide(gas6, oil6)
        X["frac_stages_per_1000ft"] = 1000 * _divide(stages, length)
        X["log_proppant_per_stage_lbs"] = np.log1p(_divide(X["Proppant_LBS"], stages).clip(lower=0))
        X["log_fluid_per_stage_gal"] = np.log1p(_divide(X["TotalWaterPumped_GAL"], stages).clip(lower=0))
        return X
