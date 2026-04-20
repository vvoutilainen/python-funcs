import numpy as np
from statsmodels.stats.weightstats import DescrStatsW

def f_mean(ignore_na=False):
    """Mean function for aggregation.

    Parameters
    ----------
    ignore_na : bool, default False
        If False, return NaN when any value is NaN. If True, drop NaN
        values and compute the mean over the remainder.

    Returns
    -------
    callable
        A function suitable for groupby.agg on a Series.
    """
    def mean_(x):
        if ignore_na:
            x = x.dropna()
        elif x.isna().any():
            return np.nan
        if x.empty:
            return np.nan
        return x.mean()
    mean_.__name__ = "mean"
    return mean_

def f_quantile(q, ignore_na=False, method="averaged_inverted_cdf"):
    """Quantile function for aggregation.

    Defaults to the inverse-empirical-CDF definition (method
    'averaged_inverted_cdf'), matching statsmodels.DescrStatsW.quantile.
    With equal weights this produces the same result as f_wquantile.

    Parameters
    ----------
    q : float
        Quantile [0, 1].
    ignore_na : bool, default False
        If False, return NaN when any value is NaN. If True, drop NaN
        values and compute the quantile over the remainder.
    method : str, default 'averaged_inverted_cdf'
        Passed through to numpy.quantile. See its docs for alternatives
        (e.g. 'linear' for the classical interpolated definition).

    Returns
    -------
    callable
        A function suitable for groupby.agg on a Series.
    """
    def quantile_(x):
        if ignore_na:
            x = x.dropna()
        elif x.isna().any():
            return np.nan
        if x.empty:
            return np.nan
        return np.quantile(x.values, q, method=method)
    quantile_.__name__ = "quantile_%s" % q
    return quantile_

def f_wmean(weights, ignore_na=False):
    """Weighted mean function for aggregation.

    Parameters
    ----------
    weights : pd.Series
        A series of weights, index-aligned to the original data frame.
    ignore_na : bool, default False
        If False, return NaN when any value or weight in the group is NaN.
        If True, drop rows where either is NaN and compute over the remainder.

    Returns
    -------
    callable
        A function suitable for groupby.agg on a Series.
    """
    def wmean_(x):
        w = weights.loc[x.index]
        if ignore_na:
            mask = x.notna() & w.notna()
            x = x[mask]
            w = w[mask]
        else:
            if x.isna().any() or w.isna().any():
                return np.nan
        # If left with no observations, return nan
        if x.empty:
            return np.nan
        # Check that weights sum to above-zero value. If not, return nan.
        if w.sum() <= 0:
            return np.nan

        # Calculate weighted mean
        return np.average(x.values, weights=w.values)

    wmean_.__name__ = "wmean"
    return wmean_

def f_wquantile(q, weights, ignore_na=False):
    """Weighted quantile function for aggregation.

    Parameters
    ----------
    q : float
        Quantile [0, 1].
    weights : pd.Series
        A series of weights, index-aligned to the original data frame.
    ignore_na : bool, default False
        If False, return NaN when any value or weight in the group is NaN.
        If True, drop rows where either is NaN and compute over the remainder.

    Returns
    -------
    callable
        A function suitable for groupby.agg on a Series.
    """
    def wquantile_(x):
        w = weights.loc[x.index]
        if ignore_na:
            mask = x.notna() & w.notna()
            x = x[mask]
            w = w[mask]
        else:
            if x.isna().any() or w.isna().any():
                return np.nan
        # If left with no observations, return nan
        if x.empty:
            return np.nan
        # Check that weights sum to above-zero value. If not, return nan.
        if w.sum() <= 0:
            return np.nan

        # Calculate weighted quantile. For some reason, if length of x or
        # weights is one, the weighted calculation fails. Cannot understand;
        # why not just return that value? Do this manually.
        if len(x) == 1:
            return x.values[0]
        else:
            return (
                DescrStatsW(
                    x.values,
                    weights=w.values
                )
                .quantile(
                    q,
                    return_pandas=False,
                )[0]
            )
    wquantile_.__name__ = "wquantile_%s" % q
    return wquantile_
