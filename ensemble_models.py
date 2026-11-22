from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

GERMAN_NOMINAL_FEATURES = (
    "Account_Balance",
    "Payment_Status_of_Previous_Credit",
    "Purpose",
    "Value_Savings_Stocks",
    "Length_of_current_employment",
    "Sex_Marital_Status",
    "Guarantors",
    "Most_valuable_available_asset",
    "Concurrent_Credits",
    "Type_of_apartment",
    "Occupation",
    "Telephone",
    "Foreign_Worker",
)
GERMAN_NOMINAL_INDICES = (0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19)
GERMAN_NUMERIC_FEATURES = (
    "Duration_of_Credit_monthly",
    "Credit_Amount",
    "Age_years",
    "Instalment_per_cent",
    "Duration_in_Current_address",
    "No_of_Credits_at_this_Bank",
    "No_of_dependents",
)
GERMAN_NUMERIC_INDICES = (1, 4, 7, 10, 12, 15, 17)
SELECTED_MODEL_PARAMETERS = {
    "n_estimators": 500,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "class_weight": "balanced",
}
SELECTED_MODEL_NAME = (
    "Random Forest (500 trees, min_samples_leaf=2; mixed feature preprocessing)"
)


def _make_pipeline(features, estimator, feature_encoding="mixed"):
    feature_count = features.shape[1] if hasattr(features, "shape") else len(features[0])

    if feature_encoding == "one_hot_all":
        preprocessor = ColumnTransformer(
            [("categorical", OneHotEncoder(handle_unknown="ignore"), list(range(feature_count)))]
        )
    elif feature_encoding == "mixed":
        columns = getattr(features, "columns", None)
        if columns is not None and set(
            GERMAN_NOMINAL_FEATURES + GERMAN_NUMERIC_FEATURES
        ).issubset(columns):
            nominal_columns = list(GERMAN_NOMINAL_FEATURES)
            numeric_columns = list(GERMAN_NUMERIC_FEATURES)
        elif feature_count == 20:
            nominal_columns = list(GERMAN_NOMINAL_INDICES)
            numeric_columns = list(GERMAN_NUMERIC_INDICES)
        else:
            nominal_columns = []
            numeric_columns = []
        if nominal_columns and numeric_columns:
            preprocessor = ColumnTransformer(
                [
                    (
                        "categorical",
                        OneHotEncoder(handle_unknown="ignore"),
                        nominal_columns,
                    ),
                    ("numeric", "passthrough", numeric_columns),
                ],
            )
        else:
            preprocessor = "passthrough"
    elif feature_encoding == "numeric_codes":
        preprocessor = "passthrough"
    else:
        raise ValueError(f"Unknown feature encoding: {feature_encoding}")

    return make_pipeline(preprocessor, estimator)


def _make_estimator(family, parameters, random_seed):
    estimator_type = {
        "random_forest": RandomForestClassifier,
        "extra_trees": ExtraTreesClassifier,
    }.get(family)
    if estimator_type is None:
        raise ValueError(f"Unknown ensemble family: {family}")
    return estimator_type(random_state=random_seed, n_jobs=-1, **parameters)


def _predict_binary(pipeline, features):
    classes = np.asarray(pipeline.classes_)
    positive_label = 1 if np.any(classes == 1) else classes[-1]
    positive_index = int(np.flatnonzero(classes == positive_label)[0])
    probabilities = np.asarray(pipeline.predict_proba(features)[:, positive_index], dtype=float)
    predictions = np.asarray(pipeline.predict(features)) == positive_label
    return probabilities, predictions.astype(int), positive_label


def _calculate_metrics(target, probabilities, predictions, positive_label):
    binary_target = (np.asarray(target) == positive_label).astype(int)
    return {
        "roc_auc": float(roc_auc_score(binary_target, probabilities)),
        "accuracy": float(accuracy_score(binary_target, predictions)),
        "precision": float(precision_score(binary_target, predictions, zero_division=0)),
        "recall": float(recall_score(binary_target, predictions, zero_division=0)),
        "f1": float(f1_score(binary_target, predictions, zero_division=0)),
    }


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    train_target = np.asarray(data.train_target)
    test_target = np.asarray(data.test_target)
    pipeline = _make_pipeline(
        data.train_features,
        _make_estimator("random_forest", SELECTED_MODEL_PARAMETERS, data.random_seed),
    )
    pipeline.fit(data.train_features, train_target)
    probabilities, predictions, positive_label = _predict_binary(
        pipeline, data.test_features
    )
    metrics = _calculate_metrics(
        test_target, probabilities, predictions, positive_label
    )

    return EnsembleResult(
        model_name=SELECTED_MODEL_NAME,
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=metrics["roc_auc"],
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
    )
