"""Checks for the metric helpers in scripts/investigate_ranking.py."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from investigate_ranking import endpoint_and_ade, spearman_rows  # noqa: E402


def test_endpoint_and_ade_hand_computed():
    gt = np.zeros((1, 3, 2))
    cands = np.zeros((1, 2, 3, 2))
    cands[0, 0, :, 0] = [1.0, 2.0, 3.0]          # off by 1, 2, 3 m
    cands[0, 1, :, 1] = [0.0, 0.0, 4.0]          # exact, then 4 m at the end
    fde, ade, last = endpoint_and_ade(cands, gt, np.ones((1, 3), bool))
    assert fde[0].tolist() == pytest.approx([3.0, 4.0])
    assert ade[0].tolist() == pytest.approx([2.0, 4.0 / 3])
    assert last.tolist() == [2]


def test_endpoint_uses_last_valid_step():
    gt = np.zeros((1, 3, 2))
    cands = np.zeros((1, 1, 3, 2))
    cands[0, 0, :, 0] = [1.0, 2.0, 100.0]        # step 3 is padding
    valid = np.array([[True, True, False]])
    fde, ade, last = endpoint_and_ade(cands, gt, valid)
    assert last.tolist() == [1]
    assert fde[0, 0] == pytest.approx(2.0)
    assert ade[0, 0] == pytest.approx(1.5)


def test_spearman_rows():
    a = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    assert spearman_rows(a, a) == pytest.approx(1.0)
    assert spearman_rows(a, -a) == pytest.approx(-1.0)
    # rows without variation are skipped instead of dividing by zero
    assert spearman_rows(np.vstack([a[:1], np.ones((1, 4))]), np.vstack([a[:1], a[:1]])) == pytest.approx(1.0)
