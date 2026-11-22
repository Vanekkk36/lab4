from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from ensemble_models import (
    SELECTED_MODEL_NAME,
    _calculate_metrics,
    _make_estimator,
    _make_pipeline,
    _predict_binary,
    run_ensemble_experiment,
)
from grader_contracts.ensembles import EnsembleInput


ROOT = Path(__file__).resolve().parent
SEED = 42
TEST_SIZE = 0.2
FOLDS = 5

CANDIDATES = [
    {
        "name": "RF mixed, sqrt, leaf=1",
        "family": "random_forest",
        "feature_encoding": "mixed",
        "parameters": {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 1, "class_weight": "balanced"},
    },
    {
        "name": "RF mixed, sqrt, leaf=2",
        "family": "random_forest",
        "feature_encoding": "mixed",
        "parameters": {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 2, "class_weight": "balanced"},
    },
    {
        "name": "RF mixed, sqrt, leaf=3",
        "family": "random_forest",
        "feature_encoding": "mixed",
        "parameters": {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 3, "class_weight": "balanced"},
    },
    {
        "name": "RF mixed, sqrt, leaf=5",
        "family": "random_forest",
        "feature_encoding": "mixed",
        "parameters": {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 5, "class_weight": "balanced"},
    },
    {
        "name": "RF mixed, max_features=0.5, leaf=2",
        "family": "random_forest",
        "feature_encoding": "mixed",
        "parameters": {"n_estimators": 500, "max_features": 0.5, "min_samples_leaf": 2, "class_weight": "balanced"},
    },
    {
        "name": "RF mixed, max_features=0.7, leaf=2",
        "family": "random_forest",
        "feature_encoding": "mixed",
        "parameters": {"n_estimators": 500, "max_features": 0.7, "min_samples_leaf": 2, "class_weight": "balanced"},
    },
    {
        "name": "RF mixed, max_features=None, leaf=2",
        "family": "random_forest",
        "feature_encoding": "mixed",
        "parameters": {"n_estimators": 500, "max_features": None, "min_samples_leaf": 2, "class_weight": "balanced"},
    },
    {
        "name": "Extra Trees mixed, sqrt, leaf=1",
        "family": "extra_trees",
        "feature_encoding": "mixed",
        "parameters": {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 1, "class_weight": "balanced"},
    },
    {
        "name": "Extra Trees mixed, sqrt, leaf=2",
        "family": "extra_trees",
        "feature_encoding": "mixed",
        "parameters": {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 2, "class_weight": "balanced"},
    },
    {
        "name": "RF one-hot all features, sqrt, leaf=2",
        "family": "random_forest",
        "feature_encoding": "one_hot_all",
        "parameters": {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 2, "class_weight": "balanced"},
    },
    {
        "name": "RF numeric codes, sqrt, leaf=2",
        "family": "random_forest",
        "feature_encoding": "numeric_codes",
        "parameters": {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 2, "class_weight": "balanced"},
    },
]


def main():
    dataset = pd.read_csv(ROOT / "german.csv", sep=";")
    features = dataset.drop(columns=["Creditability"])
    target = dataset["Creditability"]
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=SEED,
    )
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    selected_probabilities = None
    selected_predictions = None
    for candidate in CANDIDATES:
        estimator = _make_estimator(
            candidate["family"], candidate["parameters"], SEED
        )
        pipeline = _make_pipeline(
            train_features, estimator, candidate["feature_encoding"]
        )
        scores = cross_val_score(
            pipeline,
            train_features,
            train_target,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
        )
        candidate["cv_fold_roc_auc"] = [float(score) for score in scores]
        candidate["train_cv_roc_auc"] = float(scores.mean())

    selected = max(CANDIDATES, key=lambda candidate: candidate["train_cv_roc_auc"])

    for candidate in CANDIDATES:
        estimator = _make_estimator(
            candidate["family"], candidate["parameters"], SEED
        )
        pipeline = _make_pipeline(
            train_features, estimator, candidate["feature_encoding"]
        )
        pipeline.fit(train_features, train_target)
        probabilities, predictions, positive_label = _predict_binary(
            pipeline, test_features
        )
        candidate["test_metrics"] = _calculate_metrics(
            test_target, probabilities, predictions, positive_label
        )
        if candidate is selected:
            selected_probabilities = probabilities
            selected_predictions = predictions

    if selected["name"] != "RF mixed, sqrt, leaf=2":
        raise RuntimeError(
            f"CV selected {selected['name']}, but the contract model is {SELECTED_MODEL_NAME}"
        )

    contract_result = run_ensemble_experiment(
        EnsembleInput(
            train_features,
            train_target,
            test_features,
            test_target,
            random_seed=SEED,
        )
    )
    contract_metrics = {
        "roc_auc": contract_result.roc_auc,
        "accuracy": contract_result.accuracy,
        "precision": contract_result.precision,
        "recall": contract_result.recall,
        "f1": contract_result.f1,
    }
    if not np.allclose(
        list(contract_metrics.values()),
        list(selected["test_metrics"].values()),
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError("Contract metrics do not match the selected experiment")
    if not np.array_equal(contract_result.predictions, selected_predictions):
        raise RuntimeError("Contract predictions do not match the selected experiment")
    if not np.allclose(
        contract_result.probabilities, selected_probabilities, rtol=0, atol=1e-12
    ):
        raise RuntimeError("Contract probabilities do not match the selected experiment")

    results = {
        "environment": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "dataset": {
            "file": "german.csv",
            "rows": int(len(dataset)),
            "features": int(features.shape[1]),
            "target": "Creditability",
            "target_counts": {
                str(label): int(count)
                for label, count in target.value_counts().sort_index().items()
            },
            "missing_values": int(dataset.isna().sum().sum()),
        },
        "split": {
            "train_size": int(len(train_target)),
            "test_size": int(len(test_target)),
            "test_fraction": TEST_SIZE,
            "stratified": True,
            "random_seed": SEED,
        },
        "cross_validation": {
            "strategy": "StratifiedKFold",
            "n_splits": FOLDS,
            "shuffle": True,
            "random_seed": SEED,
        },
        "selection_criterion": "maximum mean 5-fold train ROC-AUC",
        "selected_model": SELECTED_MODEL_NAME,
        "selected_parameters": {
            "family": selected["family"],
            "feature_encoding": selected["feature_encoding"],
            **selected["parameters"],
            "random_seed": SEED,
        },
        "train_cv_roc_auc": selected["train_cv_roc_auc"],
        "test_metrics": selected["test_metrics"],
        "contract_check": "passed: outputs match selected CV experiment",
        "experiments": [
            {
                "name": candidate["name"],
                "family": candidate["family"],
                "feature_encoding": candidate["feature_encoding"],
                "parameters": candidate["parameters"],
                "cv_fold_roc_auc": candidate["cv_fold_roc_auc"],
                "train_cv_roc_auc": candidate["train_cv_roc_auc"],
                "test_metrics": candidate["test_metrics"],
            }
            for candidate in CANDIDATES
        ],
    }
    (ROOT / "experiment_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Selected by train CV: {selected['name']}")
    print(f"Mean train CV ROC-AUC: {selected['train_cv_roc_auc']:.5f}")
    print("Experiment | CV ROC-AUC | Test ROC-AUC | Accuracy | Precision | Recall | F1")
    for candidate in CANDIDATES:
        metrics = candidate["test_metrics"]
        print(
            f"{candidate['name']} | "
            f"{candidate['train_cv_roc_auc']:.5f} | "
            f"{metrics['roc_auc']:.5f} | "
            f"{metrics['accuracy']:.5f} | "
            f"{metrics['precision']:.5f} | "
            f"{metrics['recall']:.5f} | "
            f"{metrics['f1']:.5f}"
        )


if __name__ == "__main__":
    main()
