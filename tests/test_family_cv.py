"""Family-grouped folds."""

import pandas as pd
import pytest

from family_cv import family_folds

BIPIA_FAMILIES = {"A": 6, "B": 4, "C": 3, "D": 2, "E": 1, "F": 1}


def frame():
    """BIPIA twins (one clean, one attacked per group) in six families, plus ten
    PoisonedRAG groups with one family and two benign rows each."""
    rows, i = [], 0
    for family, n in BIPIA_FAMILIES.items():
        for _ in range(n):
            rows.append({"source": "bipia", "group": f"b{i}", "label": 0, "attack_family": None, "text": f"clean {i}"})
            rows.append({"source": "bipia", "group": f"b{i}", "label": 1, "attack_family": family, "text": f"attack {i}"})
            i += 1
    for q in range(10):
        for label in (0, 0, 1):
            rows.append({"source": "poisonedrag", "group": f"p{q}", "label": label,
                         "attack_family": "poisonedrag" if label else None, "text": f"passage {q}"})
    return pd.DataFrame(rows)


def test_twins_share_a_fold():
    data = frame()
    folds = family_folds(data, k=3)
    assert data.assign(fold=folds).groupby("group").fold.nunique().max() == 1


def test_each_family_lands_in_one_fold():
    data = frame()
    attacks = data.assign(fold=family_folds(data, k=3)).query("label == 1 and source == 'bipia'")
    assert attacks.groupby("attack_family").fold.nunique().max() == 1


def test_greedy_buckets_are_deterministic():
    # Largest first into the lightest bucket, ties to the lowest index (spec decision 3).
    data = frame()
    expected = {"A": 0, "B": 1, "C": 2, "D": 2, "E": 1, "F": 1}
    for shuffled in (data, data.sample(frac=1, random_state=1).reset_index(drop=True)):
        attacks = shuffled.assign(fold=family_folds(shuffled, k=3)).query("source == 'bipia' and label == 1")
        assert attacks.groupby("attack_family").fold.first().to_dict() == expected


def test_single_family_source_is_dealt_by_group():
    data = frame()
    pr = data.assign(fold=family_folds(data, k=3)).query("source == 'poisonedrag'")
    counts = pr.groupby("group").fold.first().value_counts()
    assert sorted(counts) == [3, 3, 4]


def test_a_group_with_two_families_raises():
    data = frame()
    data.loc[len(data)] = {"source": "bipia", "group": "b0", "label": 1, "attack_family": "B", "text": "second"}
    with pytest.raises(ValueError, match="more than one attack family"):
        family_folds(data, k=3)


def test_a_fold_without_both_labels_raises():
    data = frame().query("group in ['b0', 'b1']")
    with pytest.raises(ValueError, match="both labels"):
        family_folds(data, k=3)
