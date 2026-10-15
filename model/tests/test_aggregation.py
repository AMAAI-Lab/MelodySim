import pytest

torch = pytest.importorskip("torch")

from model.aggregation import aggregate_decision_matrix


@pytest.mark.parametrize("transpose", [False, True], ids=["long-first", "short-first"])
@pytest.mark.parametrize("case,expected", [
    ("all-matching", 1),
    ("no-matches", 0),
    ("partial-both-covered", 1),
    ("long-piece-under-covered", 0),
    ("short-piece-under-covered", 0),
])
def test_unequal_window_counts_use_each_pieces_own_coverage(transpose, case, expected):
    matrix = torch.zeros(10, 6)
    if case == "all-matching":
        # The old implementation rejected even this extreme length difference.
        matrix = torch.ones(10, 2)
    elif case == "partial-both-covered":
        matrix[:5, :3] = 1  # 50% of each piece has a match.
    elif case == "long-piece-under-covered":
        matrix[:3, :5] = 1  # 30% of piece 1; the old denominator swap accepted it.
    elif case == "short-piece-under-covered":
        matrix[:8, :2] = 1  # 33% of piece 2 is insufficient despite 80% of piece 1.
    if transpose:
        matrix = matrix.T
    decision, mask = aggregate_decision_matrix(matrix, proportion_thres=0.4, decision_thres=0.99)
    assert decision == expected
    assert mask.dtype == torch.bool
    assert torch.equal(mask, matrix.bool())


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("min_hits,expected", [(1, 1), (2, 1), (3, 0)])
def test_min_hits_is_enforced_in_both_directions(transpose, min_hits, expected):
    matrix = torch.ones(10, 2)
    if transpose:
        matrix = matrix.T
    decision, _ = aggregate_decision_matrix(matrix, proportion_thres=0.4,
                                           decision_thres=0.99, min_hits=min_hits)
    assert decision == expected


def test_existing_strict_score_and_proportion_boundaries_are_preserved():
    # Scores equal to the threshold remain nonmatches.
    decision, mask = aggregate_decision_matrix(torch.full((4, 4), 0.5))
    assert decision == 0 and not mask.any()
    # Two matched windows out of five equal 0.4; strict coverage still rejects it.
    matrix = torch.zeros(5, 5)
    matrix[:2, :2] = 1
    assert aggregate_decision_matrix(matrix, proportion_thres=0.4)[0] == 0
    assert aggregate_decision_matrix(matrix, proportion_thres=0.39)[0] == 1
