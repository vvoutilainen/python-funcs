import numpy as np
import pandas as pd
import pytest

from python_funcs.merges import decorator_merge_prints, merge_left_as_base, merge_with_amounts


# --- merge_with_amounts tests ---

class TestMergePrints:
    def setup_method(self):
        self.df_left = pd.DataFrame({
            "id": ["a", "b", "c", "d"],
            "val_l": [1, 2, 3, 4],
        })
        self.df_right = pd.DataFrame({
            "id": ["a", "b", "c", "e"],
            "val_r": [10, 20, 30, 50],
        })

    def test_inner_merge(self, capsys):
        result = merge_with_amounts(self.df_left, self.df_right, on="id", how="inner")
        assert len(result) == 3
        assert "_merge" not in result.columns
        captured = capsys.readouterr()
        assert "both: 3" in captured.out

    def test_left_merge(self, capsys):
        result = merge_with_amounts(self.df_left, self.df_right, on="id", how="left")
        assert len(result) == 4
        assert "_merge" not in result.columns
        captured = capsys.readouterr()
        assert "both: 3" in captured.out
        assert "left_only: 1" in captured.out

    def test_right_merge(self, capsys):
        result = merge_with_amounts(self.df_left, self.df_right, on="id", how="right")
        assert len(result) == 4
        assert "_merge" not in result.columns
        captured = capsys.readouterr()
        assert "both: 3" in captured.out
        assert "right_only: 1" in captured.out

    def test_outer_merge(self, capsys):
        result = merge_with_amounts(self.df_left, self.df_right, on="id", how="outer")
        assert len(result) == 5
        assert "_merge" not in result.columns
        captured = capsys.readouterr()
        assert "both: 3" in captured.out
        assert "left_only: 1" in captured.out
        assert "right_only: 1" in captured.out

    def test_indicator_true_keeps_merge_column(self, capsys):
        result = merge_with_amounts(
            self.df_left, self.df_right, on="id", how="outer", indicator=True
        )
        assert "_merge" in result.columns
        capsys.readouterr()

    def test_indicator_false_drops_merge_column(self, capsys):
        result = merge_with_amounts(
            self.df_left, self.df_right, on="id", how="outer", indicator=False
        )
        assert "_merge" not in result.columns
        capsys.readouterr()

    def test_default_drops_merge_column(self, capsys):
        """Default behavior (no indicator arg) should drop _merge."""
        result = merge_with_amounts(self.df_left, self.df_right, on="id", how="inner")
        assert "_merge" not in result.columns
        capsys.readouterr()

    def test_merge_on_index(self, capsys):
        df_left = pd.DataFrame(
            {"l": [1, 2]},
            index=pd.MultiIndex.from_tuples([("a", 1), ("a", 2)]),
        )
        df_right = pd.DataFrame(
            {"r": [10, 20, 30]},
            index=pd.MultiIndex.from_tuples([("a", 1), ("b", 1), ("b", 2)]),
        )
        result = merge_with_amounts(
            df_left, df_right, left_index=True, right_index=True, how="outer"
        )
        assert len(result) == 4
        captured = capsys.readouterr()
        assert "both: 1" in captured.out

    def test_merge_empty_frames(self, capsys):
        df_left = pd.DataFrame({"id": pd.Series([], dtype="str"), "v": pd.Series([], dtype="float")})
        df_right = pd.DataFrame({"id": pd.Series([], dtype="str"), "w": pd.Series([], dtype="float")})
        result = merge_with_amounts(df_left, df_right, on="id", how="outer")
        assert len(result) == 0
        capsys.readouterr()

    def test_merge_one_side_empty(self, capsys):
        df_empty = pd.DataFrame({"id": pd.Series([], dtype="str"), "v": pd.Series([], dtype="float")})
        result = merge_with_amounts(self.df_left, df_empty, on="id", how="left")
        assert len(result) == len(self.df_left)
        captured = capsys.readouterr()
        assert "left_only: 4" in captured.out

    def test_merge_with_duplicates(self, capsys):
        """Many-to-many join should produce cartesian product for matching keys."""
        df_left = pd.DataFrame({"id": ["a", "a"], "v": [1, 2]})
        df_right = pd.DataFrame({"id": ["a", "a"], "w": [10, 20]})
        result = merge_with_amounts(df_left, df_right, on="id", how="inner")
        assert len(result) == 4
        captured = capsys.readouterr()
        assert "both: 4" in captured.out

    def test_merge_with_nan_keys(self, capsys):
        df_left = pd.DataFrame({"id": ["a", np.nan], "v": [1, 2]})
        df_right = pd.DataFrame({"id": ["a", np.nan], "w": [10, 20]})
        result = merge_with_amounts(df_left, df_right, on="id", how="outer")
        assert len(result) >= 2
        capsys.readouterr()

    def test_prints_output_format(self, capsys):
        merge_with_amounts(self.df_left, self.df_right, on="id", how="outer")
        captured = capsys.readouterr()
        assert captured.out.startswith("Appearances in merged frame:")


# --- decorator_merge_prints tests ---

class TestDecoratorMergePrints:
    def test_custom_function(self, capsys):
        @decorator_merge_prints
        def custom_merge(left, right):
            return pd.merge(left, right, on="id", how="outer", indicator=True)

        df_left = pd.DataFrame({"id": ["a", "b"], "v": [1, 2]})
        df_right = pd.DataFrame({"id": ["a", "c"], "w": [10, 30]})
        result = custom_merge(df_left, df_right)
        assert "_merge" not in result.columns
        captured = capsys.readouterr()
        assert "both: 1" in captured.out

    def test_raises_if_no_indicator(self):
        @decorator_merge_prints
        def bad_merge(left, right):
            return pd.merge(left, right, on="id", how="inner")

        df_left = pd.DataFrame({"id": ["a"], "v": [1]})
        df_right = pd.DataFrame({"id": ["a"], "w": [10]})
        with pytest.raises(ValueError, match="_merge"):
            bad_merge(df_left, df_right)

    def test_preserves_function_name(self):
        @decorator_merge_prints
        def my_special_merge(left, right):
            return pd.merge(left, right, on="id", indicator=True)

        assert my_special_merge.__name__ == "my_special_merge"

    def test_custom_func_with_indicator_kwarg(self, capsys):
        """When using decorator directly, indicator kwarg controls column retention."""
        @decorator_merge_prints
        def custom_merge(left, right, **kwargs):
            return pd.merge(left, right, on="id", how="inner", indicator=True)

        df_left = pd.DataFrame({"id": ["a", "b"], "v": [1, 2]})
        df_right = pd.DataFrame({"id": ["a", "b"], "w": [10, 20]})

        result = custom_merge(df_left, df_right, indicator=True)
        assert "_merge" in result.columns
        capsys.readouterr()

        result = custom_merge(df_left, df_right)
        assert "_merge" not in result.columns
        capsys.readouterr()


# --- merge_left_as_base tests ---

class TestMergeLeftAsBase:
    def setup_method(self):
        self.df_left = pd.DataFrame({
            "id": ["a", "b", "c", "d"],
            "val_l": [1, 2, 3, 4],
        })
        self.df_right = pd.DataFrame({
            "id": ["a", "b", "c", "e"],
            "val_r": [10, 20, 30, 50],
        })

    def test_basic_left_merge(self, capsys):
        """Left merge should keep all left rows, NaN-fill unmatched right."""
        result = merge_left_as_base(self.df_left, self.df_right, on="id")
        assert len(result) == 4
        assert "_merge" not in result.columns
        assert result.loc[result["id"] == "d", "val_r"].isna().all()
        capsys.readouterr()

    def test_prints_appearances(self, capsys):
        merge_left_as_base(self.df_left, self.df_right, on="id")
        captured = capsys.readouterr()
        assert "both: 3" in captured.out
        assert "left_only: 1" in captured.out

    def test_how_kwarg_is_ignored(self, capsys):
        """Passing how= should trigger a warning and be ignored."""
        with pytest.warns(UserWarning, match="Input 'how' is ignored"):
            result = merge_left_as_base(
                self.df_left, self.df_right, on="id", how="inner"
            )
        # Should still behave as left merge (4 rows, not 3)
        assert len(result) == 4
        capsys.readouterr()

    def test_indicator_true_keeps_merge_column(self, capsys):
        result = merge_left_as_base(
            self.df_left, self.df_right, on="id", indicator=True
        )
        assert "_merge" in result.columns
        capsys.readouterr()

    def test_indicator_false_drops_merge_column(self, capsys):
        result = merge_left_as_base(
            self.df_left, self.df_right, on="id", indicator=False
        )
        assert "_merge" not in result.columns
        capsys.readouterr()

    def test_assert_left_keys_unique(self, capsys):
        """Duplicate join keys in left frame should raise AssertionError."""
        df_left_dup = pd.DataFrame({
            "id": ["a", "a", "b"],
            "val_l": [1, 2, 3],
        })
        with pytest.raises(AssertionError, match="unique rows"):
            merge_left_as_base(df_left_dup, self.df_right, on="id")
        capsys.readouterr()

    def test_assert_output_same_unique_keys_as_left(self, capsys):
        """Right frame with duplicate join keys should inflate rows and fail assertion."""
        df_right_dup = pd.DataFrame({
            "id": ["a", "a", "b"],
            "val_r": [10, 11, 20],
        })
        with pytest.raises(AssertionError, match="same unique join key rows"):
            merge_left_as_base(self.df_left, df_right_dup, on="id")
        capsys.readouterr()

    def test_right_unique_keys_passes(self, capsys):
        """Right frame with unique join keys should not trigger assertion."""
        df_right_unique = pd.DataFrame({
            "id": ["a", "b"],
            "val_r": [10, 20],
        })
        result = merge_left_as_base(self.df_left, df_right_unique, on="id")
        assert len(result) == 4
        capsys.readouterr()

    def test_merge_on_left_on_right_on(self, capsys):
        """Should work with left_on / right_on instead of on."""
        df_left = pd.DataFrame({"key_l": ["a", "b", "c"], "v": [1, 2, 3]})
        df_right = pd.DataFrame({"key_r": ["a", "b", "d"], "w": [10, 20, 40]})
        result = merge_left_as_base(df_left, df_right, left_on="key_l", right_on="key_r")
        assert len(result) == 3
        capsys.readouterr()

    def test_merge_on_index(self, capsys):
        """Should work with left_index=True, right_index=True."""
        df_left = pd.DataFrame({"v": [1, 2]}, index=["a", "b"])
        df_right = pd.DataFrame({"w": [10, 30]}, index=["a", "c"])
        result = merge_left_as_base(df_left, df_right, left_index=True, right_index=True)
        assert len(result) == 2
        assert result.loc["b", "w"] != result.loc["b", "w"]  # NaN check
        capsys.readouterr()

    def test_merge_on_index_duplicate_left_keys_fails(self, capsys):
        """Duplicate index values in left frame should fail assertion."""
        df_left = pd.DataFrame({"v": [1, 2]}, index=["a", "a"])
        df_right = pd.DataFrame({"w": [10, 30]}, index=["a", "c"])
        with pytest.raises(AssertionError, match="unique rows"):
            merge_left_as_base(df_left, df_right, left_index=True, right_index=True)
        capsys.readouterr()

    def test_merge_on_common_columns(self, capsys):
        """When no key args given, should merge on common columns."""
        df_left = pd.DataFrame({"id": ["a", "b"], "v": [1, 2]})
        df_right = pd.DataFrame({"id": ["a", "c"], "w": [10, 30]})
        result = merge_left_as_base(df_left, df_right)
        assert len(result) == 2
        capsys.readouterr()

    def test_multikey_merge(self, capsys):
        """Should work with composite join keys."""
        df_left = pd.DataFrame({
            "k1": ["a", "a", "b"],
            "k2": [1, 2, 1],
            "v": [10, 20, 30],
        })
        df_right = pd.DataFrame({
            "k1": ["a", "b", "c"],
            "k2": [1, 1, 1],
            "w": [100, 200, 300],
        })
        result = merge_left_as_base(df_left, df_right, on=["k1", "k2"])
        assert len(result) == 3
        capsys.readouterr()

    def test_multikey_duplicate_left_fails(self, capsys):
        """Composite keys with duplicates in left should fail."""
        df_left = pd.DataFrame({
            "k1": ["a", "a", "a"],
            "k2": [1, 1, 2],
            "v": [10, 20, 30],
        })
        df_right = pd.DataFrame({
            "k1": ["a"],
            "k2": [1],
            "w": [100],
        })
        with pytest.raises(AssertionError, match="unique rows"):
            merge_left_as_base(df_left, df_right, on=["k1", "k2"])
        capsys.readouterr()

    def test_no_matching_keys(self, capsys):
        """All left_only merge should pass assertions."""
        df_right_no_match = pd.DataFrame({
            "id": ["x", "y", "z"],
            "val_r": [10, 20, 30],
        })
        result = merge_left_as_base(self.df_left, df_right_no_match, on="id")
        assert len(result) == 4
        assert result["val_r"].isna().all()
        capsys.readouterr()

    def test_empty_left_frame(self, capsys):
        """Empty left frame should produce empty result."""
        df_empty = pd.DataFrame({"id": pd.Series([], dtype="str"), "val_l": pd.Series([], dtype="float")})
        result = merge_left_as_base(df_empty, self.df_right, on="id")
        assert len(result) == 0
        capsys.readouterr()
