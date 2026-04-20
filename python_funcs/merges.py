import pandas as pd
from functools import wraps
import warnings

def decorator_merge_prints(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        drop_merge_col = not kwargs.get("indicator", False)

        df = func(*args, **kwargs)

        if "_merge" not in df.columns:
            raise ValueError(
                "Result does not contain '_merge' column. "
                "Ensure pd.merge() is called with indicator=True."
            )

        print("Appearances in merged frame: {}".format(
            ", ".join(["{}: {:,.0f}".format(k, v)
            for k, v in sorted(df.groupby("_merge").size().to_dict().items())])
        ))

        if drop_merge_col:
            df = df.drop(columns="_merge")

        return df

    return wrapper

@decorator_merge_prints
def merge_with_amounts(left, right, **kwargs):
    """Merge with prints of appearance amounts.

    Parameters
    ----------
    left : As in pandas.merge.
    right : As in pandas.merge.
    kwargs : As in pandas.merge.
    """
    kwargs.pop("indicator", None)
    return pd.merge(left, right, indicator=True, **kwargs)

@decorator_merge_prints
def merge_left_as_base(left, right, **kwargs):
    """Left-merge considering left frame join keys as forming base population.

    Parameters
    ----------
    left : As in pandas.merge.
    right : As in pandas.merge.
    kwargs : As in pandas.merge.
    """
    kwargs.pop("indicator", None)
    # If param 'how' supplied, get rid of it
    if "how" in kwargs:
        warnings.warn("Warning: Input 'how' is ignored!")
        kwargs.pop("how")

    # Determine left and right join keys
    use_index = kwargs.get("left_index", False) and kwargs.get("right_index", False)
    if use_index:
        left_keys = left.index.to_frame(index=False)
        right_keys = right.index.to_frame(index=False)
    else:
        left_key_cols = kwargs.get("left_on") or kwargs.get("on")
        right_key_cols = kwargs.get("right_on") or kwargs.get("on")
        if left_key_cols is not None:
            if isinstance(left_key_cols, str):
                left_key_cols = [left_key_cols]
            if isinstance(right_key_cols, str):
                right_key_cols = [right_key_cols]
            left_keys = left[left_key_cols]
            right_keys = right[right_key_cols].rename(
                columns=dict(zip(right_key_cols, left_key_cols))
            )
        else:
            # Default: merge on common columns
            common = left.columns.intersection(right.columns).tolist()
            left_keys = left[common]
            right_keys = right[common]

    # 1) Left join keys must form unique rows in the left frame
    assert not left_keys.duplicated().any(), (
        "Left frame join keys do not form unique rows."
    )

    # Warn if right frame has keys not present in left
    right_only_keys = right_keys.drop_duplicates().merge(
        left_keys.drop_duplicates().assign(_in_left=True),
        on=list(left_keys.columns),
        how="left",
    )
    n_right_only = right_only_keys["_in_left"].isna().sum()
    if n_right_only > 0:
        print("Warning: Right frame contains {:,} key(s) not present in left frame.".format(n_right_only))

    # Merge
    df = pd.merge(left, right, indicator=True, how="left", **kwargs)

    # 2) Exactly the same unique rows (by join keys) must be present in output
    assert left_keys.shape[0] == df.shape[0], (
        "Output frame does not contain exactly the same unique join key rows "
        "as the left frame. The right frame likely contains duplicate join keys."
    )

    return df
