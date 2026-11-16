import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_gold_geometry import (  # noqa: E402
    empirical_p,
    pair_cooccurrence,
    rowwise_spearman,
    stance_indicators,
    stance_pair_counts,
)
from scipy.stats import spearmanr  # noqa: E402

# Toy raw files: 4 sentences x 3 values; 1 = attained/constrained, 0.5/0.5 = unclear.
ATTAINED = np.array([
    [1.0, 1.0, 0.0],  # v0 attained, v1 attained         -> same (0,1)
    [1.0, 0.0, 0.0],  # v0 attained, v2 constrained      -> opposite (0,2)
    [0.0, 0.5, 0.0],  # v1 unclear, v2 constrained       -> presence (1,2), no stance pair
    [0.0, 0.0, 0.0],  # v0, v1, v2 all constrained        -> same (0,1), (0,2), (1,2)
])
CONSTRAINED = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.5, 1.0],
    [1.0, 1.0, 1.0],
])


def test_stance_indicators_split_clear_and_unclear():
    ind = stance_indicators(ATTAINED, CONSTRAINED)
    assert ind["presence"].sum() == 9
    assert ind["attained"].sum() == 3
    assert ind["constrained"].sum() == 5
    assert ind["unclear"].sum() == 1
    assert not (ind["attained"] & ind["constrained"]).any()


def test_pair_counts_on_toy_matrix():
    ind = stance_indicators(ATTAINED, CONSTRAINED)
    co = pair_cooccurrence(ind["presence"])
    assert co.tolist() == [[3, 2, 2], [2, 3, 2], [2, 2, 3]]
    same, opp = stance_pair_counts(ind["attained"], ind["constrained"])
    assert np.allclose(same, same.T) and np.allclose(opp, opp.T)
    # off-diagonal: (0,1) same twice; (0,2) opposite once, same once; (1,2) same once
    # (the unclear v1 in sentence 3 contributes no stance pair)
    assert same[0, 1] == 2 and opp[0, 1] == 0
    assert same[0, 2] == 1 and opp[0, 2] == 1
    assert same[1, 2] == 1 and opp[1, 2] == 0


def test_rowwise_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(0)
    x = rng.integers(1, 5, size=(6, 20))
    y = rng.normal(size=20)
    expected = [spearmanr(row, y).correlation for row in x]
    assert np.allclose(rowwise_spearman(x, y), expected)


def test_empirical_p_counts_observed_as_one_draw():
    null = np.array([0.0, 1.0, 2.0, 3.0])
    assert empirical_p(2.0, null, "greater") == (1 + 2) / 5
    assert empirical_p(2.0, null, "less") == (1 + 3) / 5
