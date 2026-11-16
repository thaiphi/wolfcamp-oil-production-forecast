"""Dataset preparation and retrospective cohorts for the Wolfcamp month-6 task."""
from pathlib import Path
import numpy as np
import pandas as pd

TARGET = "oil_months_7_to_12_bbl"
CUMULATIVE_6 = "First6MonthOil_BBL"
CUMULATIVE_12 = "First12MonthOil_BBL"
NUMERIC_FEATURES = [
    "First3MonthOil_BBL", "First3MonthGas_MCF", "First3MonthWater_BBL",
    "First6MonthOil_BBL", "First6MonthGas_MCF", "First6MonthWater_BBL",
    "LateralLength_FT", "TVD_FT", "MD_FT", "Proppant_LBS",
    "ProppantIntensity_LBSPerFT", "TotalWaterPumped_GAL",
    "WaterIntensity_GALPerFT", "FracStages", "CompletionTime_DAYS",
    "SoakTime_DAYS", "ENVElevationKB_FT", "FirstProdYear",
]
CATEGORICAL_FEATURES = [
    "StateProvince", "County", "ENVSubPlay", "ENVFluidType",
    "ENVFracJobType", "ENVProppantType", "ENVWellboreType",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
SOURCE_COLUMNS = ["CompletionID", "WellID", "FirstProdDate", CUMULATIVE_12] + FEATURES
DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data" / "raw" / "wolfcamp_wells.csv"


def load_data(path=DEFAULT_DATA):
    """Return one valid record per well, ordered by FirstProdDate/CompletionID.

    No target-based encoding, scaling, or imputation is done before the split.
    The source has repeat completions for some wells, so use one completion
    per WellID to keep a well from straddling holdouts. This ordering does not
    independently establish completion chronology or historical data availability.
    """
    frame = pd.read_csv(path, low_memory=False, usecols=SOURCE_COLUMNS)
    for name in NUMERIC_FEATURES + [CUMULATIVE_12]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["FirstProdDate"] = pd.to_datetime(frame["FirstProdDate"], errors="coerce")
    frame = frame.dropna(subset=["WellID", "FirstProdDate", "FirstProdYear", CUMULATIVE_6, CUMULATIVE_12])
    frame = frame[(frame[CUMULATIVE_6] >= 0) & (frame[CUMULATIVE_12] >= frame[CUMULATIVE_6])]
    frame = frame.sort_values(["FirstProdDate", "CompletionID"]).drop_duplicates("WellID", keep="first")
    frame[TARGET] = frame[CUMULATIVE_12] - frame[CUMULATIVE_6]
    frame = frame.sort_values(["FirstProdDate", "WellID"]).reset_index(drop=True)
    assert frame["WellID"].is_unique
    assert (frame[TARGET] >= 0).all()
    return frame


def split_by_year(frame):
    """Chronological holdouts. 2014-18 training, 2019 validation, 2020-21 test."""
    train = frame.loc[frame.FirstProdYear.between(2014, 2018)].copy()
    valid = frame.loc[frame.FirstProdYear == 2019].copy()
    test = frame.loc[frame.FirstProdYear.between(2020, 2021)].copy()
    if min(map(len, (train, valid, test))) < 10:
        raise ValueError("Not enough records in chronological train/val/test cohorts")
    for a, b in [(train, valid), (train, test), (valid, test)]:
        assert set(a.WellID).isdisjoint(b.WellID)
    return train, valid, test


def get_xy(frame):
    # Never feed a future production target or its derived feature into X.
    return frame.loc[:, FEATURES].copy(), frame[TARGET].to_numpy(dtype=float)
