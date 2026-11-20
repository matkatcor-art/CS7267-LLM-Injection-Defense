"""Family-grouped cross-validation for choosing a detector's settings on attack families it
was not fit on. Spec: docs/superpowers/specs/2026-11-08-b1-family-cv-design.md."""

import numpy as np


def family_folds(frame, k=5, seed=7267) -> np.ndarray:
    """One fold in [0, k) per row. A group never spans two folds. In a source with several
    attack families, each family goes whole into one fold: largest first into the fold with
    the fewest attack rows, ties to the lowest index. Other groups (one-family sources, groups
    with no attack) are shuffled with `seed` and dealt round-robin."""
    attacks = frame[frame.label == 1]
    per_group = attacks.groupby("group").attack_family.nunique()
    if (per_group > 1).any():
        raise ValueError(f"groups with more than one attack family: {list(per_group[per_group > 1].index[:5])}")

    group_fold = {}
    for source, rows in attacks.groupby("source"):
        sizes = rows.attack_family.value_counts()
        if len(sizes) < 2:
            continue
        load, family_fold = [0] * k, {}
        for family in sorted(sizes.index, key=lambda f: (-sizes[f], f)):
            target = min(range(k), key=lambda i: (load[i], i))
            family_fold[family] = target
            load[target] += int(sizes[family])
        group_fold.update({g: family_fold[f] for g, f in rows.groupby("group").attack_family.first().items()})

    rest = sorted(set(frame.group) - set(group_fold))
    order = np.random.default_rng(seed).permutation(len(rest))
    group_fold.update({rest[j]: n % k for n, j in enumerate(order)})

    folds = frame.group.map(group_fold).to_numpy(dtype=int)
    labels = frame.label.to_numpy()
    for f in range(k):
        if len(set(labels[folds == f])) < 2:
            raise ValueError(f"fold {f} needs both labels; use fewer folds or more groups")
    return folds
