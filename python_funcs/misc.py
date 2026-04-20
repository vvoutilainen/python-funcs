import numpy as np

def pct_change_fallback(x1, x2, eps=1e-9, fallback="symmetric", zero_value=0.0):
    """Percentage change with a fallback to symmetric midpoint/arc change.

    Parameters
    ----------
    x1 : array-like
        Start (baseline) values.
    x2 : array-like
        End values.
    eps : float, optional
        Threshold for "tiny" baseline values.
    fallback : str, optional
        Fallback type when |x1| is near zero. Defaults to symmetric
        midpoint/arc change. Accepts also setting near-zero baseline cases to NA.
    zero_value : float
        Replacement value when both baseline and end value are zero.

    Returns
    -------
    float or numpy.ndarray
        Relative change as a decimal (e.g. 0.10 for a 10% increase).
        Scalar input returns float; array input returns ndarray.
    """
    assert fallback in ["symmetric", "nan"], (
        "Invalid fallback! Accepted values: 'symmetric', 'nan'."
    )
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    # Standard % change (|x1| as denominator so negative baselines are handled
    # correctly)
    std = (x2 - x1) / np.abs(x1)

    # Symmetric (midpoint / arc) % change as fallback for near-zero baselines
    den = np.abs(x1) + np.abs(x2)
    sym = np.where(den > 0.0, 2.0 * (x2 - x1) / den, zero_value)

    # "Safe" condition: baseline is non-tiny (positive or negative)
    use_std = np.abs(x1) > eps
    if fallback == "symmetric":
        out = np.where(use_std, std, sym)
    else:
        out = np.where(use_std, std, np.nan)

    return out.item() if out.shape == () else out
