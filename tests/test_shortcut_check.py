"""The shortcut check runs end to end on the fixture dataset and returns a tri-state verdict."""
import shortcut_check
from shortcut_probe import bootstrap_ci
import numpy as np
from sklearn.metrics import roc_auc_score


def test_check_runs_on_fixture(dataset_path, tmp_path):
    report = shortcut_check.main(["--detector", "tfidf", "--data", str(dataset_path), "--out", str(tmp_path / "o")])
    assert report.verdict in {"PASS", "FAIL", "INCONCLUSIVE"}
    assert (tmp_path / "o" / "verdict.txt").read_text().startswith(report.verdict)
    assert set(report.gate_table.eval_set) == {"test", "mcptox", "msb"}


def test_group_bootstrap_ci_contains_point():
    rng = np.random.default_rng(0)
    y = np.tile([0, 1], 100)
    s = y + rng.normal(0, 1, 200)
    g = np.repeat(np.arange(50), 4)
    point, lo, hi = bootstrap_ci(roc_auc_score, y, s, g)
    assert lo <= point <= hi
