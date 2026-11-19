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
    suggest, trial_params, trials_table,
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


def test_trials_table_is_the_same_from_memory_and_from_csv(dataset_path, tmp_path):
    study, _ = run_study(train_rows(dataset_path), n_trials=6, progress=False)
    raw = tmp_path / "raw.csv"
    study.trials_dataframe().to_csv(raw, index=False)  # the format the first full run saved
    from_memory = trials_table(study.trials_dataframe())
    from_csv = trials_table(pd.read_csv(raw))
    pd.testing.assert_frame_equal(from_memory.drop(columns="duration_s"),
                                  from_csv.drop(columns="duration_s"), check_dtype=False)
    for name in ("max_features", "class_weight"):
        assert from_memory[name].notna().all()  # None is written as "none", never left empty


def test_trial_params_drop_unused_settings_and_keep_none():
    row = {"number": 0, "state": "COMPLETE", "value": 0.9, "duration_s": 1.0, "C": 1.0,
           "features": "word", "word_max_n": 2, "char_range": float("nan"), "max_features": "none",
           "auroc_fold0": 0.9}
    assert trial_params(row) == {"C": 1.0, "features": "word", "word_max_n": 2, "max_features": "none"}


def runs(uri):
    import mlflow

    mlflow.set_tracking_uri(uri)
    return mlflow.search_runs(experiment_names=["phase3-detectors"])


def test_log_only_uploads_the_saved_study_with_the_same_threshold(dataset_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a sqlite tracking store puts artifacts in ./mlruns
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    out_dir, out = tmp_path / "study", tmp_path / "report.csv"
    common = ["--data", str(dataset_path), "--out-dir", str(out_dir), "--out", str(out),
              "--tracking-uri", uri]
    main(common + ["--n-trials", "3", "--n-boot", "10"])
    main(common + ["--log-only"])
    logged = runs(uri)
    parents = logged[logged["tags.mlflow.runName"] == "tfidf-optuna"]
    trials = logged[logged["tags.mlflow.runName"].str.startswith("trial-")]
    assert len(parents) == 2 and len(trials) == 6
    assert parents["params.threshold"].nunique() == 1
    assert parents["params.best_trial"].nunique() == 1
    assert set(trials["tags.mlflow.parentRunId"]) == set(parents.run_id)


def test_a_failed_upload_keeps_the_files_and_says_how_to_retry(dataset_path, tmp_path, monkeypatch):
    import tfidf_optuna

    def broken(*args, **kwargs):
        raise ConnectionError("503")

    monkeypatch.setattr(tfidf_optuna, "log_study", broken)
    out_dir, out = tmp_path / "study", tmp_path / "report.csv"
    with pytest.raises(SystemExit, match="ARGS=--log-only"):
        main(["--data", str(dataset_path), "--n-trials", "2", "--n-boot", "10", "--out-dir",
              str(out_dir), "--out", str(out), "--tracking-uri", "http://127.0.0.1:1"])
    assert (out_dir / "best_params.json").exists() and out.exists()
