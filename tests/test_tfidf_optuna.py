"""B1-Optuna: the configurable detector, the search space, the holdout guard and the study."""

import json
import re

import numpy as np
import optuna
import pandas as pd
import pytest

from dataset import HoldoutError, load
from harness import load_data
from tfidf import TfidfDetector
from tfidf_optuna import (
    CHAR_RANGES, TunedTfidfDetector, check_fit_rows, detector_params, main, progress_line, run_study,
    suggest,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_rows(dataset_path):
    return load("fit", path=dataset_path, roles=["train_pool"], splits=["train"]).to_pandas()


def test_defaults_reproduce_b1(dataset_path):
    data = load_data(dataset_path)
    b1 = TfidfDetector(C=10).fit(data.train.text, data.train.label)
    tuned = TunedTfidfDetector(C=10).fit(data.train.text, data.train.label)
    np.testing.assert_allclose(tuned.predict_proba(data.val.text), b1.predict_proba(data.val.text))


@pytest.mark.parametrize("features", ["word", "char", "both"])
def test_every_feature_set_fits(dataset_path, features):
    data = load_data(dataset_path)
    model = TunedTfidfDetector(features=features, char_analyzer="char", char_range=(2, 4),
                               word_max_n=3, class_weight="balanced", max_features=50)
    scores = model.fit(data.train.text, data.train.label).predict_proba(data.val.text)
    assert scores.shape == (len(data.val),)
    assert model.top_features(3)["adversarial"]


def test_unknown_feature_set_raises():
    with pytest.raises(ValueError, match="word, char or both"):
        TunedTfidfDetector(features="bytes")


def test_suggest_samples_only_the_settings_its_features_use():
    for features, absent in (("word", {"char_analyzer", "char_range"}), ("char", {"word_max_n"})):
        trial = optuna.trial.FixedTrial({"C": 1.0, "features": features, "word_max_n": 2,
                                         "char_analyzer": "char", "char_range": "3-6", "min_df": 1,
                                         "max_features": None, "sublinear_tf": True,
                                         "class_weight": None})
        params = suggest(trial)
        assert absent.isdisjoint(params)
        assert set(trial.params) == set(params)
    assert detector_params({"char_range": "3-6"}) == {"char_range": CHAR_RANGES["3-6"]}


@pytest.mark.parametrize("split, role", [("val", "train_pool"), ("test", "train_pool"),
                                         ("test", "loso_holdout"), ("test", "transfer_holdout")])
def test_the_study_refuses_rows_outside_train_pool_train(dataset_path, split, role):
    frame = train_rows(dataset_path)
    frame.loc[0, ["split", "role"]] = [split, role]
    with pytest.raises(HoldoutError):
        check_fit_rows(frame)
    with pytest.raises(HoldoutError):
        run_study(frame, n_trials=1)


def test_a_fixed_seed_gives_the_same_best_trial(dataset_path):
    frame = train_rows(dataset_path)
    first, folds = run_study(frame, n_trials=4)
    second, _ = run_study(frame, n_trials=4)
    assert first.best_trial.number == second.best_trial.number
    assert first.best_params == second.best_params
    assert first.best_value == second.best_value
    assert folds.max() == 4
    assert all(f"auroc_fold{f}" in first.best_trial.user_attrs for f in range(5))


def test_each_trial_prints_a_progress_line(dataset_path, capsys):
    run_study(train_rows(dataset_path), n_trials=3)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("trial ")]
    assert [line.split()[1] for line in lines] == ["1/3", "2/3", "3/3"]
    pattern = re.compile(r"trial \d/3  COMPLETE [01]\.\d{4} +best [01]\.\d{4} \(#\d\) +\d+\.\ds$")
    assert all(pattern.fullmatch(line) for line in lines), lines


def test_progress_can_be_turned_off(dataset_path, capsys):
    run_study(train_rows(dataset_path), n_trials=2, progress=False)
    assert "trial " not in capsys.readouterr().out


def test_a_pruned_trial_says_how_many_folds_it_ran():
    study = optuna.create_study(direction="maximize")
    study.add_trial(optuna.trial.create_trial(params={}, distributions={}, value=0.9))
    pruned = optuna.trial.create_trial(state=optuna.trial.TrialState.PRUNED, params={},
                                       distributions={}, user_attrs={"auroc_fold0": 0.5,
                                                                     "auroc_fold1": 0.6})
    study.add_trial(pruned)
    line = progress_line(study, study.trials[1], 100)
    assert line.startswith("trial   2/100  PRUNED after 2 folds")
    assert "best 0.9000 (#0)" in line


def test_main_writes_the_study_and_the_report(dataset_path, tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    out_dir, out = tmp_path / "study", tmp_path / "b1_tfidf_lr_optuna.csv"
    study, report = main(["--data", str(dataset_path), "--n-trials", "3", "--n-boot", "10",
                          "--out-dir", str(out_dir), "--out", str(out)])
    best = json.loads((out_dir / "best_params.json").read_text())
    assert best["params"] == study.best_params
    assert len(pd.read_csv(out_dir / "trials.csv")) == 3
    assert (out_dir / "history.png").exists()
    written = pd.read_csv(out)
    assert set(written.detector) == {"tfidf-optuna"}
    assert set(written.eval_set) == {"test", "mcptox", "msb"}
    assert list(written.columns) == list(report.columns)
