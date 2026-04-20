import numpy as np
import pytest

from python_funcs.misc import pct_change_fallback


class TestPctChangeFallback:
    def test_standard_case(self):
        """100 -> 110 is a 0.10 increase."""
        result = pct_change_fallback(100, 110)
        assert result == pytest.approx(0.10)

    def test_standard_case_array(self):
        result = pct_change_fallback([100, 200], [110, 250])
        np.testing.assert_allclose(result, [0.10, 0.25])

    def test_zero_baseline_symmetric_fallback(self):
        """Zero baseline triggers symmetric formula: 2*(5-0)/(0+5) = 2.0."""
        result = pct_change_fallback(0, 5)
        assert result == pytest.approx(2.0)

    def test_both_zero(self):
        """Both zero returns zero_value (default 0.0)."""
        result = pct_change_fallback(0, 0)
        assert result == pytest.approx(0.0)

    def test_both_zero_custom_value(self):
        result = pct_change_fallback(0, 0, zero_value=-9.99)
        assert result == pytest.approx(-9.99)

    def test_negative_baseline_uses_standard(self):
        """Negative baseline uses standard formula with |x1| denominator."""
        result = pct_change_fallback(-5, 10)
        # standard: (10-(-5))/5 = 3.0
        assert result == pytest.approx(3.0)

    def test_negative_baseline_negative_end(self):
        """Negative baseline to less-negative end."""
        result = pct_change_fallback(-4, -2)
        # standard: (-2-(-4))/4 = 0.5
        assert result == pytest.approx(0.5)

    def test_negative_baseline_positive_end_distinguishable(self):
        """Two different positive ends from same negative baseline give different results."""
        assert pct_change_fallback(-4, 2) == pytest.approx(1.5)
        assert pct_change_fallback(-4, 5) == pytest.approx(2.25)

    def test_tiny_baseline_below_eps(self):
        """Baseline smaller than eps triggers fallback."""
        result = pct_change_fallback(1e-12, 1.0)
        # symmetric: 2*(1-1e-12)/(1e-12+1) ≈ 2.0
        assert result == pytest.approx(2.0, rel=1e-6)

    def test_fallback_nan_mode(self):
        """Non-standard cases return NaN when fallback='nan'."""
        result = pct_change_fallback(0, 5, fallback="nan")
        assert np.isnan(result)

    def test_fallback_nan_standard_case_unaffected(self):
        """Standard cases still work normally with fallback='nan'."""
        result = pct_change_fallback(100, 110, fallback="nan")
        assert result == pytest.approx(0.10)

    def test_scalar_returns_float(self):
        """Scalar inputs should return a Python float, not an ndarray."""
        result = pct_change_fallback(100, 110)
        assert isinstance(result, float)

    def test_array_returns_ndarray(self):
        """Array inputs should return an ndarray."""
        result = pct_change_fallback([100, 200], [110, 250])
        assert isinstance(result, np.ndarray)

    def test_decrease(self):
        """200 -> 150 is a -0.25 change."""
        result = pct_change_fallback(200, 150)
        assert result == pytest.approx(-0.25)

    def test_positive_baseline_negative_end_uses_standard(self):
        """Positive x1 with negative x2 uses standard formula, not symmetric."""
        result = pct_change_fallback(10, -4)
        # standard: (-4 - 10) / 10 = -1.4
        assert result == pytest.approx(-1.4)

    def test_positive_baseline_negative_end_distinguishable(self):
        """Two inputs with same x2 but different x1 should give different results."""
        assert pct_change_fallback(10, -4) != pytest.approx(pct_change_fallback(3, -4))
