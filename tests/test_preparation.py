import numpy as np
import pandas as pd
import pytest

from python_funcs.preparation import (
    remove_nas,
    remove_above_thr,
    remove_zero_and_below,
    remove_given_vals,
    keep_given_vals,
    limit_to_range,
)


@pytest.fixture
def sample_df():
    """DataFrame with a mix of values for testing filter functions."""
    return pd.DataFrame({
        "value": [np.nan, -1.0, 0.0, 5.0, 10.0, 20.0, 50.0],
        "amount": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0],
    })


class TestRemoveNas:
    def test_removes_nan_rows(self, sample_df):
        result = remove_nas(sample_df, "value")
        assert result["value"].isna().sum() == 0
        assert len(result) == 6

    def test_no_nans_unchanged(self):
        df = pd.DataFrame({"value": [1.0, 2.0], "amount": [10.0, 20.0]})
        result = remove_nas(df, "value")
        assert len(result) == 2


class TestRemoveAboveThr:
    def test_removes_above_threshold(self, sample_df):
        result = remove_above_thr(sample_df, "value", thr=10.0)
        assert (result["value"].dropna() <= 10.0).all()

    def test_threshold_is_inclusive(self, sample_df):
        result = remove_above_thr(sample_df, "value", thr=10.0)
        assert 10.0 in result["value"].values


class TestRemoveZeroAndBelow:
    def test_removes_zero_and_negative(self, sample_df):
        result = remove_zero_and_below(sample_df, "value")
        assert (result["value"] > 0).all()
        # Original has NaN, -1, 0 that should be removed (NaN fails > 0)
        assert len(result) == 4  # 5, 10, 20, 50


class TestRemoveGivenVals:
    def test_removes_specified_values(self, sample_df):
        result = remove_given_vals(sample_df, "value", values=[5.0, 10.0])
        assert 5.0 not in result["value"].values
        assert 10.0 not in result["value"].values

    def test_other_values_preserved(self, sample_df):
        result = remove_given_vals(sample_df, "value", values=[5.0])
        assert 20.0 in result["value"].values
        assert 50.0 in result["value"].values


class TestKeepGivenVals:
    def test_keeps_only_specified(self, sample_df):
        result = keep_given_vals(sample_df, "value", values=[5.0, 10.0])
        assert set(result["value"].values) == {5.0, 10.0}

    def test_length_correct(self, sample_df):
        result = keep_given_vals(sample_df, "value", values=[5.0, 10.0])
        assert len(result) == 2


class TestLimitToRange:
    def test_limits_to_range(self, sample_df):
        result = limit_to_range(sample_df, "value", range_start=0.0, range_end=20.0)
        assert (result["value"] >= 0.0).all()
        assert (result["value"] <= 20.0).all()

    def test_range_is_inclusive(self, sample_df):
        result = limit_to_range(sample_df, "value", range_start=5.0, range_end=20.0)
        assert 5.0 in result["value"].values
        assert 20.0 in result["value"].values


class TestDecoratorOutput:
    def test_prints_function_name_and_variable(self, sample_df, capsys):
        remove_nas(sample_df, "value")
        captured = capsys.readouterr()
        assert "remove_nas" in captured.out
        assert "value" in captured.out

    def test_prints_observation_loss(self, sample_df, capsys):
        remove_nas(sample_df, "value")
        captured = capsys.readouterr()
        assert "Lost 1 (14.3%) observations" in captured.out

    def test_prints_volume_loss_with_amount_col(self, sample_df, capsys):
        remove_nas(sample_df, "value", amount_col="amount")
        captured = capsys.readouterr()
        # Row 0 (amount=100) is removed: 100 of 2800 = 3.6%
        assert "Lost 1 (14.3%) observations, 100 (3.6%) in amount" in captured.out

    def test_no_volume_without_amount_col(self, sample_df, capsys):
        remove_nas(sample_df, "value")
        captured = capsys.readouterr()
        assert "in amount" not in captured.out

    def test_functools_wraps_preserves_name(self):
        assert remove_nas.__name__ == "remove_nas"
        assert "Remove NAs" in remove_nas.__doc__
