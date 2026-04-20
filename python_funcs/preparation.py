import functools
import pandas as pd

def length_prints(func):
    """Decorator to print amounts lost to transformations."""
    @functools.wraps(func)
    def wrapper(df_in, variable, *args, amount_col=None, **kwargs):
        # Print what we are doing
        print("\n{} for variable {}:".format(func.__name__, variable))

        # Get original amounts
        orig_len = len(df_in)
        if amount_col is not None:
            orig_vol = df_in[amount_col].sum()

        # Apply limitations to frame
        df_in = func(df_in, variable, *args, **kwargs)

        # Print how many observations were changed
        lost_len = orig_len - len(df_in)
        lost_len_pct = lost_len / orig_len if orig_len > 0 else 0
        if amount_col is not None:
            lost_vol = orig_vol - df_in[amount_col].sum()
            lost_vol_pct = lost_vol / orig_vol if orig_vol != 0 else 0
            print(
                (
                    "Lost {:,.0f} ({:.1%}) observations, {:,.0f} ({:.1%}) "
                    "in {}."
                ).format(
                    lost_len,
                    lost_len_pct,
                    lost_vol,
                    lost_vol_pct,
                amount_col
            ))
        else:
            print("Lost {:,.0f} ({:.1%}) observations.".format(
                lost_len,
                lost_len_pct
            ))

        return df_in
    return wrapper

@length_prints
def remove_nas(df_in, variable):
    """Remove NAs in certain column"""
    df_in = df_in[~pd.isnull(df_in[variable])]
    return df_in

@length_prints
def remove_above_thr(df_in, variable, **kwargs):
    """Remove values above certain threshold in given column"""
    thr = kwargs.get("thr", None)
    if thr is None:
        raise ValueError("'thr' must be provided for remove_above_thr.")
    print("Threshold is {:.4f}".format(thr))
    df_in = df_in[df_in[variable] <= thr]
    return df_in

@length_prints
def remove_zero_and_below(df_in, variable):
    """Remove values at and below zero in given column"""
    df_in = df_in[df_in[variable] > 0]
    return df_in

@length_prints
def remove_given_vals(df_in, variable, values):
    """Remove observations with given values."""
    print("Excluded values are:")
    print(*values, sep='\n')
    df_in = df_in[~df_in[variable].isin(values)]
    return df_in

@length_prints
def keep_given_vals(df_in, variable, values):
    """Keep observations with given values in column."""
    print("Kept values are:")
    print(*values, sep='\n')
    df_in = df_in[df_in[variable].isin(values)]
    return df_in

@length_prints
def limit_to_range(df_in, variable, **kwargs):
    """Limit to given period range in given column"""
    range_start = kwargs.get("range_start", None)
    range_end = kwargs.get("range_end", None)
    df_in = df_in[(df_in[variable] >= range_start) & (df_in[variable] <= range_end)]
    return df_in
