"""
Titanic Feature Engineering Pipeline Experiment
- Dataset: Titanic - Machine Learning from Disaster (Kaggle-style train.csv)
- Task: Binary classification, Survived prediction
- Requirements covered: EDA, missing-value strategies, encoding strategies,
  scaling strategies, derived features, feature selection, model comparison,
  Pipeline, GridSearchCV, feature importance visualization.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "train.csv"
FIG_DIR = BASE_DIR / "figures"
RESULT_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def mutual_info_fixed(X, y):
    """Mutual information selector with fixed random_state for reproducibility."""
    return mutual_info_classif(X, y, random_state=RANDOM_STATE)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load local train.csv. If local file is absent, download from a public mirror."""
    if path.exists():
        return pd.read_csv(path)
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-informed features for Titanic survival prediction."""
    df = df.copy()

    # 1) Family features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"].replace(0, 1)

    # 2) Title from passenger name
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_map = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Rare", "Countess": "Rare", "Capt": "Rare", "Col": "Rare",
        "Don": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare",
        "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare",
    }
    df["Title"] = df["Title"].replace(title_map).fillna("Unknown")

    # 3) Cabin-derived features
    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    df["Deck"] = df["Cabin"].astype(str).str[0].replace("n", np.nan)

    # 4) Age group; missing values remain missing and are handled by imputers.
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 19, 35, 60, 100],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
        include_lowest=True,
    )

    return df


def get_feature_sets():
    numeric_features = [
        "Pclass", "Age", "SibSp", "Parch", "Fare",
        "FamilySize", "IsAlone", "FarePerPerson", "CabinKnown",
    ]
    categorical_features = ["Sex", "Embarked", "Title", "Deck", "AgeGroup"]
    base_features = ["Pclass", "SibSp", "Parch", "Fare"]
    return numeric_features, categorical_features, base_features


def make_eda_figures(df: pd.DataFrame) -> dict[str, str]:
    """Create mandatory EDA visualizations."""
    paths = {}

    # Missing ratio
    missing_ratio = df.isna().mean().sort_values(ascending=False) * 100
    missing_ratio.to_csv(RESULT_DIR / "missing_ratio.csv", header=["missing_ratio_percent"])
    plt.figure(figsize=(8, 4.5))
    plt.barh(missing_ratio.index, missing_ratio.values)
    plt.gca().invert_yaxis()
    plt.title("Missing Value Ratio by Column")
    plt.xlabel("Missing ratio (%)")
    plt.ylabel("Column")
    plt.tight_layout()
    path = FIG_DIR / "eda_missing_ratio.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths["missing_ratio"] = str(path)

    # Target distribution countplot
    plt.figure(figsize=(5.5, 4))
    df["Survived"].value_counts().sort_index().plot(kind="bar")
    plt.title("Target Distribution: Survived")
    plt.xlabel("Survived (0=No, 1=Yes)")
    plt.ylabel("Count")
    plt.tight_layout()
    path = FIG_DIR / "eda_target_countplot.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths["target_countplot"] = str(path)

    # Histogram - Age and Fare
    plt.figure(figsize=(7, 4.2))
    sns.histplot(data=df, x="Age", hue="Survived", kde=True, bins=30)
    plt.title("Age Distribution by Survival")
    plt.tight_layout()
    path = FIG_DIR / "eda_age_histogram.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths["age_histogram"] = str(path)

    plt.figure(figsize=(7, 4.2))
    sns.histplot(data=df, x="Fare", hue="Survived", kde=True, bins=35)
    plt.title("Fare Distribution by Survival")
    plt.tight_layout()
    path = FIG_DIR / "eda_fare_histogram.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths["fare_histogram"] = str(path)

    # Boxplot - outlier exploration
    plt.figure(figsize=(6, 4.2))
    sns.boxplot(data=df, x="Survived", y="Fare")
    plt.title("Fare Outlier Check by Survival")
    plt.tight_layout()
    path = FIG_DIR / "eda_fare_boxplot.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths["fare_boxplot"] = str(path)

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=False)
    plt.title("Correlation Heatmap of Numeric Variables")
    plt.tight_layout()
    path = FIG_DIR / "eda_correlation_heatmap.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths["correlation_heatmap"] = str(path)

    return paths


def make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    missing_strategy: str,
    encoding: str,
    scaling: str | None,
) -> ColumnTransformer:
    """Build preprocessing pipeline according to an experimental setting."""
    if missing_strategy == "mean":
        num_imputer = SimpleImputer(strategy="mean")
        cat_imputer = SimpleImputer(strategy="most_frequent")
    elif missing_strategy == "median":
        num_imputer = SimpleImputer(strategy="median")
        cat_imputer = SimpleImputer(strategy="most_frequent")
    elif missing_strategy == "most_frequent":
        num_imputer = SimpleImputer(strategy="most_frequent")
        cat_imputer = SimpleImputer(strategy="most_frequent")
    elif missing_strategy == "none":
        # Base experiment uses only columns without missing values.
        num_imputer = "passthrough"
        cat_imputer = "drop"
    else:
        raise ValueError(f"Unknown missing_strategy: {missing_strategy}")

    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    elif scaling == "robust":
        scaler = RobustScaler()
    elif scaling is None:
        scaler = "passthrough"
    else:
        raise ValueError(f"Unknown scaling: {scaling}")

    numeric_pipe = Pipeline([
        ("imputer", num_imputer),
        ("scaler", scaler),
    ])

    if encoding == "onehot":
        categorical_pipe = Pipeline([
            ("imputer", cat_imputer),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
    elif encoding == "label":
        # For multiple categorical columns, OrdinalEncoder is used as a pipeline-safe Label Encoding variant.
        categorical_pipe = Pipeline([
            ("imputer", cat_imputer),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
    elif encoding == "none":
        categorical_pipe = "drop"
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return feature names after ColumnTransformer fitting."""
    try:
        names = preprocessor.get_feature_names_out()
        return [name.replace("num__", "").replace("cat__", "") for name in names]
    except Exception:
        return [f"feature_{i}" for i in range(preprocessor.transformers_[0][1].shape[1])]


def evaluate_pipeline(pipe: Pipeline, X_train, X_test, y_train, y_test) -> dict:
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
    else:
        proba = pipe.decision_function(X_test)

    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, proba),
        "ConfusionMatrix": confusion_matrix(y_test, pred).tolist(),
    }


def run_experiments(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    df_fe = add_derived_features(df)
    numeric_features, categorical_features, base_features = get_feature_sets()

    y = df_fe["Survived"]
    X_full = df_fe[numeric_features + categorical_features]
    X_base = df_fe[base_features]

    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train_base, X_test_base, _, _ = train_test_split(
        X_base, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    experiments = {
        "Base": {
            "X_train": X_train_base,
            "X_test": X_test_base,
            "numeric": base_features,
            "categorical": [],
            "missing": "none",
            "encoding": "none",
            "scaling": None,
            "feature_selection": False,
            "description": "No advanced FE; numeric complete columns only",
        },
        "Exp-1": {
            "X_train": X_train_full,
            "X_test": X_test_full,
            "numeric": numeric_features,
            "categorical": categorical_features,
            "missing": "mean",
            "encoding": "onehot",
            "scaling": "standard",
            "feature_selection": False,
            "description": "Mean imputation + One-Hot + StandardScaler",
        },
        "Exp-2": {
            "X_train": X_train_full,
            "X_test": X_test_full,
            "numeric": numeric_features,
            "categorical": categorical_features,
            "missing": "median",
            "encoding": "label",
            "scaling": "minmax",
            "feature_selection": True,
            "description": "Median imputation + Label/Ordinal encoding + MinMaxScaler + SelectKBest",
        },
        "Exp-3": {
            "X_train": X_train_full,
            "X_test": X_test_full,
            "numeric": numeric_features,
            "categorical": categorical_features,
            "missing": "most_frequent",
            "encoding": "onehot",
            "scaling": "robust",
            "feature_selection": True,
            "description": "Most frequent imputation + One-Hot + RobustScaler + SelectKBest",
        },
    }

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE, max_depth=None),
    }

    rows = []
    fitted_pipelines = {}

    for exp_name, cfg in experiments.items():
        preprocessor = make_preprocessor(
            cfg["numeric"], cfg["categorical"], cfg["missing"], cfg["encoding"], cfg["scaling"]
        )
        for model_name, model in models.items():
            steps = [("preprocess", preprocessor)]
            if cfg["feature_selection"]:
                steps.append(("selector", SelectKBest(score_func=mutual_info_fixed, k=10)))
            steps.append(("clf", clone(model)))
            pipe = Pipeline(steps)

            metrics = evaluate_pipeline(pipe, cfg["X_train"], cfg["X_test"], y_train, y_test)
            row = {
                "Experiment": exp_name,
                "Model": model_name,
                "Missing": cfg["missing"],
                "Encoding": cfg["encoding"],
                "Scaling": cfg["scaling"] or "none",
                "FeatureSelection": "O" if cfg["feature_selection"] else "X",
                "Description": cfg["description"],
            }
            row.update({k: v for k, v in metrics.items() if k != "ConfusionMatrix"})
            rows.append(row)
            fitted_pipelines[(exp_name, model_name)] = pipe

    results = pd.DataFrame(rows).sort_values(["ROC_AUC", "F1"], ascending=False)
    results.to_csv(RESULT_DIR / "experiments_summary.csv", index=False)

    # GridSearchCV bonus: use Exp-3 + RandomForest pipeline because it has robust preprocessing and feature selection.
    cfg = experiments["Exp-3"]
    gs_pipe = Pipeline([
        ("preprocess", make_preprocessor(cfg["numeric"], cfg["categorical"], cfg["missing"], cfg["encoding"], cfg["scaling"])),
        ("selector", SelectKBest(score_func=mutual_info_fixed, k=10)),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])
    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [3, 5],
        "clf__min_samples_split": [2, 5],
    }
    grid = GridSearchCV(gs_pipe, param_grid=param_grid, cv=3, scoring="roc_auc", n_jobs=1)
    grid.fit(cfg["X_train"], y_train)
    grid_pred = grid.predict(cfg["X_test"])
    grid_proba = grid.predict_proba(cfg["X_test"])[:, 1]
    grid_result = pd.DataFrame([{
        "Experiment": "Exp-3 + GridSearchCV",
        "Model": "RandomForest",
        "BestParams": str(grid.best_params_),
        "CV_Best_ROC_AUC": grid.best_score_,
        "Test_Accuracy": accuracy_score(y_test, grid_pred),
        "Test_Precision": precision_score(y_test, grid_pred),
        "Test_Recall": recall_score(y_test, grid_pred),
        "Test_F1": f1_score(y_test, grid_pred),
        "Test_ROC_AUC": roc_auc_score(y_test, grid_proba),
    }])
    grid_result.to_csv(RESULT_DIR / "grid_search_result.csv", index=False)

    # Feature importance visualization: use fitted Exp-1 RandomForest, before feature selection.
    imp_pipe = fitted_pipelines[("Exp-1", "RandomForest")]
    preprocessor = imp_pipe.named_steps["preprocess"]
    feature_names = get_feature_names(preprocessor)
    importances = imp_pipe.named_steps["clf"].feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.to_csv(RESULT_DIR / "feature_importance_top.csv", index=False)

    plt.figure(figsize=(8, 5))
    top15 = importance_df.head(15).iloc[::-1]
    plt.barh(top15["feature"], top15["importance"])
    plt.title("Random Forest Feature Importance - Top 15")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance_top15.png", dpi=160)
    plt.close()

    # Feature selection comparison table: Exp-1 vs Exp-3 One-Hot based RF/LR summary.
    fs_compare = results[results["Experiment"].isin(["Exp-1", "Exp-3"])].copy()
    fs_compare.to_csv(RESULT_DIR / "feature_selection_compare.csv", index=False)

    return results, fitted_pipelines, grid_result, importance_df


def make_basic_tables(df: pd.DataFrame) -> None:
    column_desc = pd.DataFrame([
        ["PassengerId", "Passenger identifier", "Identifier"],
        ["Survived", "Survival status: 0 = No, 1 = Yes", "Target"],
        ["Pclass", "Ticket class: 1/2/3", "Ordinal categorical / numeric"],
        ["Name", "Passenger name", "Text"],
        ["Sex", "Passenger sex", "Categorical"],
        ["Age", "Passenger age", "Numeric"],
        ["SibSp", "Number of siblings/spouses aboard", "Numeric"],
        ["Parch", "Number of parents/children aboard", "Numeric"],
        ["Ticket", "Ticket number", "Text"],
        ["Fare", "Passenger fare", "Numeric"],
        ["Cabin", "Cabin number", "Categorical/Text"],
        ["Embarked", "Port of embarkation", "Categorical"],
    ], columns=["Column", "Description", "Type"])
    column_desc.to_csv(RESULT_DIR / "column_description.csv", index=False)

    pd.DataFrame({
        "rows": [df.shape[0]],
        "columns": [df.shape[1]],
        "target": ["Survived"],
        "positive_class_rate": [df["Survived"].mean()],
    }).to_csv(RESULT_DIR / "dataset_shape_summary.csv", index=False)


def main():
    df = load_data()
    print(f"Loaded Titanic dataset: shape={df.shape}")
    make_basic_tables(df)
    make_eda_figures(df)
    results, fitted_pipelines, grid_result, importance_df = run_experiments(df)
    print("\n=== Experiment Results ===")
    print(results[["Experiment", "Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]].to_string(index=False))
    print("\n=== GridSearch Result ===")
    print(grid_result.to_string(index=False))
    print("\n=== Top Features ===")
    print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
