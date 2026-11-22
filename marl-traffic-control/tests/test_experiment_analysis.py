import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_results import assessment, checked_matrix, holm


def test_paired_assessment_zero_and_constant_effects():
    zero = assessment(np.zeros((5, 10)), 1000)
    assert zero["demand_lo"] == zero["crossed_hi"] == 0
    assert zero["conditional_signflip_p"] == 1
    positive = assessment(np.full((5, 10), 3.0), 1000)
    assert positive["crossed_lo"] == positive["crossed_hi"] == 3
    assert positive["conditional_signflip_p"] == 2 / 1024
    assert positive["n_init"] == 5 and positive["n_demand"] == 10


def test_crossed_bootstrap_recovers_initialization_variance():
    deltas = np.repeat(np.arange(5)[:, None], 10, axis=1)
    result = assessment(deltas, 1000)
    assert result["demand_lo"] == result["demand_hi"] == 2
    assert result["crossed_lo"] < 2 < result["crossed_hi"]
    assert result == assessment(deltas, 1000)


def test_holm_preserves_order_and_monotonicity():
    assert holm([0.04, 0.002, 0.03, 0.9]) == pytest.approx([0.09, 0.008, 0.09, 0.9])


def test_incomplete_matrix_fails_before_statistics(tmp_path):
    task = (tmp_path, tmp_path / "route.xml", "fixed-10", None, None, 600, "test", "C1", 2000)
    with pytest.raises(ValueError, match="1 missing"):
        checked_matrix(tmp_path, {}, [task])
    with pytest.raises(ValueError, match="Invalid paired matrix"):
        assessment([[np.nan]], 100)
