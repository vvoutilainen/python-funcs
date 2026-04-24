import numpy as np
import pandas as pd
import pytest

from python_funcs.aggregation import f_mean, f_quantile, f_wmean, f_wquantile, f_wsum


# --- f_mean tests ---

class TestFMean:
    def test_simple_mean(self):
        values = pd.Series([2.0, 4.0, 6.0])
        assert f_mean()(values) == pytest.approx(4.0)

    def test_single_observation(self):
        values = pd.Series([42.0])
        assert f_mean()(values) == pytest.approx(42.0)

    def test_nan_returns_nan_by_default(self):
        values = pd.Series([1.0, 2.0, np.nan, 4.0])
        assert np.isnan(f_mean()(values))

    def test_ignore_na_drops_nans(self):
        values = pd.Series([1.0, 2.0, np.nan, 4.0])
        assert f_mean(ignore_na=True)(values) == pytest.approx(7.0 / 3)

    def test_all_nan(self):
        values = pd.Series([np.nan, np.nan])
        assert np.isnan(f_mean(ignore_na=True)(values))

    def test_callable_name(self):
        assert f_mean().__name__ == "mean"


# --- f_quantile tests ---

class TestFQuantile:
    def test_median_odd_length(self):
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        agg_fn = f_quantile(0.5)
        assert agg_fn(values) == pytest.approx(3.0)

    def test_median_even_length(self):
        values = pd.Series([1.0, 2.0, 3.0, 4.0])
        agg_fn = f_quantile(0.5)
        assert agg_fn(values) == pytest.approx(2.5)

    def test_quartile(self):
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        agg_fn = f_quantile(0.25)
        assert agg_fn(values) == pytest.approx(2.0)

    def test_nan_returns_nan_by_default(self):
        values = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        agg_fn = f_quantile(0.5)
        assert np.isnan(agg_fn(values))

    def test_ignore_na_drops_nans(self):
        values = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        agg_fn = f_quantile(0.5, ignore_na=True)
        assert agg_fn(values) == pytest.approx(3.0)

    def test_all_nan(self):
        values = pd.Series([np.nan, np.nan])
        agg_fn = f_quantile(0.5, ignore_na=True)
        assert np.isnan(agg_fn(values))

    def test_single_observation(self):
        values = pd.Series([42.0])
        agg_fn = f_quantile(0.5)
        assert agg_fn(values) == pytest.approx(42.0)

    def test_kwargs_forwarded_midpoint(self):
        """interpolation='midpoint' differs from default on even lengths."""
        values = pd.Series([1.0, 2.0, 3.0, 4.0])
        agg_fn = f_quantile(0.25, method="midpoint")
        # midpoint between 1 and 2 = 1.5 (vs default 1.75)
        assert agg_fn(values) == pytest.approx(1.5)

    def test_kwargs_forwarded_lower(self):
        values = pd.Series([1.0, 2.0, 3.0, 4.0])
        agg_fn = f_quantile(0.5, method="lower")
        assert agg_fn(values) == pytest.approx(2.0)

    def test_callable_name(self):
        agg_fn = f_quantile(0.75)
        assert agg_fn.__name__ == "quantile_0.75"

    @pytest.mark.parametrize("values", [
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [10.0, 20.0, 30.0],
        [5.0, 8.0],
    ])
    @pytest.mark.parametrize("q", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_matches_wquantile_with_equal_weights(self, values, q):
        """f_quantile should equal f_wquantile when all weights are equal."""
        s = pd.Series(values, index=range(len(values)))
        w = pd.Series([1.0] * len(values), index=range(len(values)))
        assert f_quantile(q)(s) == pytest.approx(f_wquantile(q, w)(s))


# --- f_wmean tests ---

class TestFWmean:
    def test_equal_weights(self):
        """With equal weights, weighted mean equals simple mean."""
        values = pd.Series([2.0, 4.0, 6.0], index=[0, 1, 2])
        weights = pd.Series([1.0, 1.0, 1.0], index=[0, 1, 2])
        agg_fn = f_wmean(weights)
        assert agg_fn(values) == pytest.approx(4.0)

    def test_unequal_weights(self):
        """Weight concentrated on first observation."""
        values = pd.Series([2.0, 4.0, 6.0], index=[0, 1, 2])
        weights = pd.Series([1.0, 0.0, 0.0], index=[0, 1, 2])
        agg_fn = f_wmean(weights)
        assert agg_fn(values) == pytest.approx(2.0)

    def test_known_weighted_mean(self):
        values = pd.Series([10.0, 20.0], index=[0, 1])
        weights = pd.Series([3.0, 1.0], index=[0, 1])
        agg_fn = f_wmean(weights)
        # (10*3 + 20*1) / (3+1) = 50/4 = 12.5
        assert agg_fn(values) == pytest.approx(12.5)

    def test_all_nan(self):
        values = pd.Series([np.nan, np.nan], index=[0, 1])
        weights = pd.Series([1.0, 1.0], index=[0, 1])
        agg_fn = f_wmean(weights, ignore_na=True)
        assert np.isnan(agg_fn(values))

    def test_nan_value_returns_nan_by_default(self):
        values = pd.Series([np.nan, 10.0, 20.0], index=[0, 1, 2])
        weights = pd.Series([5.0, 1.0, 1.0], index=[0, 1, 2])
        agg_fn = f_wmean(weights)
        assert np.isnan(agg_fn(values))

    def test_nan_weight_returns_nan_by_default(self):
        values = pd.Series([10.0, 20.0, 30.0], index=[0, 1, 2])
        weights = pd.Series([1.0, np.nan, 1.0], index=[0, 1, 2])
        agg_fn = f_wmean(weights)
        assert np.isnan(agg_fn(values))

    def test_ignore_na_drops_nan_values(self):
        """NaN values are dropped; weights re-align by index."""
        values = pd.Series([np.nan, 10.0, 20.0], index=[0, 1, 2])
        weights = pd.Series([5.0, 1.0, 1.0], index=[0, 1, 2])
        agg_fn = f_wmean(weights, ignore_na=True)
        # After dropping index 0, remaining: values=[10, 20], weights=[1, 1]
        assert agg_fn(values) == pytest.approx(15.0)

    def test_ignore_na_drops_nan_weights(self):
        values = pd.Series([10.0, 20.0, 30.0], index=[0, 1, 2])
        weights = pd.Series([1.0, np.nan, 1.0], index=[0, 1, 2])
        agg_fn = f_wmean(weights, ignore_na=True)
        # After dropping index 1, equal weights on [10, 30]
        assert agg_fn(values) == pytest.approx(20.0)

    def test_zero_sum_weights(self):
        values = pd.Series([1.0, 2.0], index=[0, 1])
        weights = pd.Series([0.0, 0.0], index=[0, 1])
        agg_fn = f_wmean(weights)
        assert np.isnan(agg_fn(values))

    def test_single_observation(self):
        values = pd.Series([42.0], index=[0])
        weights = pd.Series([1.0], index=[0])
        agg_fn = f_wmean(weights)
        assert agg_fn(values) == pytest.approx(42.0)

    def test_callable_name(self):
        weights = pd.Series([1.0])
        agg_fn = f_wmean(weights)
        assert agg_fn.__name__ == "wmean"


# --- f_wsum tests ---

class TestFWsum:
    def test_equal_weights(self):
        values = pd.Series([2.0, 4.0, 6.0], index=[0, 1, 2])
        weights = pd.Series([1.0, 1.0, 1.0], index=[0, 1, 2])
        agg_fn = f_wsum(weights)
        assert agg_fn(values) == pytest.approx(12.0)

    def test_unequal_weights(self):
        values = pd.Series([2.0, 4.0, 6.0], index=[0, 1, 2])
        weights = pd.Series([1.0, 0.0, 0.0], index=[0, 1, 2])
        agg_fn = f_wsum(weights)
        assert agg_fn(values) == pytest.approx(2.0)

    def test_known_weighted_sum(self):
        values = pd.Series([10.0, 20.0], index=[0, 1])
        weights = pd.Series([3.0, 1.0], index=[0, 1])
        agg_fn = f_wsum(weights)
        # 10*3 + 20*1 = 50
        assert agg_fn(values) == pytest.approx(50.0)

    def test_all_nan(self):
        values = pd.Series([np.nan, np.nan], index=[0, 1])
        weights = pd.Series([1.0, 1.0], index=[0, 1])
        agg_fn = f_wsum(weights, ignore_na=True)
        assert np.isnan(agg_fn(values))

    def test_nan_value_returns_nan_by_default(self):
        values = pd.Series([np.nan, 10.0, 20.0], index=[0, 1, 2])
        weights = pd.Series([5.0, 1.0, 1.0], index=[0, 1, 2])
        agg_fn = f_wsum(weights)
        assert np.isnan(agg_fn(values))

    def test_nan_weight_returns_nan_by_default(self):
        values = pd.Series([10.0, 20.0, 30.0], index=[0, 1, 2])
        weights = pd.Series([1.0, np.nan, 1.0], index=[0, 1, 2])
        agg_fn = f_wsum(weights)
        assert np.isnan(agg_fn(values))

    def test_ignore_na_drops_nan_values(self):
        values = pd.Series([np.nan, 10.0, 20.0], index=[0, 1, 2])
        weights = pd.Series([5.0, 2.0, 3.0], index=[0, 1, 2])
        agg_fn = f_wsum(weights, ignore_na=True)
        # After dropping index 0: 10*2 + 20*3 = 80
        assert agg_fn(values) == pytest.approx(80.0)

    def test_ignore_na_drops_nan_weights(self):
        values = pd.Series([10.0, 20.0, 30.0], index=[0, 1, 2])
        weights = pd.Series([1.0, np.nan, 2.0], index=[0, 1, 2])
        agg_fn = f_wsum(weights, ignore_na=True)
        # After dropping index 1: 10*1 + 30*2 = 70
        assert agg_fn(values) == pytest.approx(70.0)

    def test_zero_sum_weights(self):
        values = pd.Series([1.0, 2.0], index=[0, 1])
        weights = pd.Series([0.0, 0.0], index=[0, 1])
        agg_fn = f_wsum(weights)
        assert np.isnan(agg_fn(values))

    def test_single_observation(self):
        values = pd.Series([42.0], index=[0])
        weights = pd.Series([2.0], index=[0])
        agg_fn = f_wsum(weights)
        assert agg_fn(values) == pytest.approx(84.0)

    def test_callable_name(self):
        weights = pd.Series([1.0])
        agg_fn = f_wsum(weights)
        assert agg_fn.__name__ == "wsum"


# --- f_wquantile tests ---

class TestFWquantile:
    def test_median_equal_weights(self):
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=range(5))
        weights = pd.Series([1.0] * 5, index=range(5))
        agg_fn = f_wquantile(0.5, weights)
        assert agg_fn(values) == pytest.approx(3.0)

    def test_all_nan(self):
        values = pd.Series([np.nan, np.nan], index=[0, 1])
        weights = pd.Series([1.0, 1.0], index=[0, 1])
        agg_fn = f_wquantile(0.5, weights, ignore_na=True)
        assert np.isnan(agg_fn(values))

    def test_nan_value_returns_nan_by_default(self):
        values = pd.Series([np.nan, 10.0, 20.0], index=[0, 1, 2])
        weights = pd.Series([1.0, 1.0, 1.0], index=[0, 1, 2])
        agg_fn = f_wquantile(0.5, weights)
        assert np.isnan(agg_fn(values))

    def test_nan_weight_returns_nan_by_default(self):
        values = pd.Series([10.0, 20.0, 30.0], index=[0, 1, 2])
        weights = pd.Series([1.0, np.nan, 1.0], index=[0, 1, 2])
        agg_fn = f_wquantile(0.5, weights)
        assert np.isnan(agg_fn(values))

    def test_zero_sum_weights(self):
        values = pd.Series([1.0, 2.0], index=[0, 1])
        weights = pd.Series([0.0, 0.0], index=[0, 1])
        agg_fn = f_wquantile(0.5, weights)
        assert np.isnan(agg_fn(values))

    def test_single_observation(self):
        """Single observation triggers the len(x)==1 special case."""
        values = pd.Series([99.0], index=[0])
        weights = pd.Series([1.0], index=[0])
        agg_fn = f_wquantile(0.5, weights)
        assert agg_fn(values) == pytest.approx(99.0)

    def test_callable_name(self):
        weights = pd.Series([1.0])
        agg_fn = f_wquantile(0.25, weights)
        assert agg_fn.__name__ == "wquantile_0.25"

    def test_ignore_na_drops_with_index_alignment(self):
        """After dropping NaN, weights align correctly by index."""
        values = pd.Series([np.nan, 10.0, 20.0], index=[0, 1, 2])
        weights = pd.Series([100.0, 1.0, 1.0], index=[0, 1, 2])
        agg_fn = f_wquantile(0.5, weights, ignore_na=True)
        # After dropping index 0, equal weights on [10, 20]
        result = agg_fn(values)
        assert 10.0 <= result <= 20.0
