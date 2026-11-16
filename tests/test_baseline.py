import numpy as np
from src.wolfcamp.modeling import RecentRateBaseline
from src.wolfcamp.data import load_data, split_by_year, get_xy


def test_recent_rate_model_learns_only_training_scale():
    train, valid, _ = split_by_year(load_data())
    x_train, y_train = get_xy(train)
    x_valid, _ = get_xy(valid)
    model = RecentRateBaseline().fit(x_train, y_train)
    assert 0 < model.scale_ < 2
    assert (model.predict(x_valid) >= 0).all()
    assert np.isfinite(model.predict(x_valid)).all()
