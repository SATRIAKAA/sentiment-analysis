from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


MODEL_NAMES = ("SVM", "Naive Bayes", "Linear Regression")
SENTIMENT_ORDER = ("Negatif", "Netral", "Positif")
SENTIMENT_ALIASES = {
    "negatif": "Negatif",
    "negative": "Negatif",
    "neg": "Negatif",
    "netral": "Netral",
    "neutral": "Netral",
    "neu": "Netral",
    "positif": "Positif",
    "positive": "Positif",
    "pos": "Positif",
}


class OrdinalLinearRegressionClassifier(ClassifierMixin, BaseEstimator):
    """Convert regression scores back to ordered sentiment classes."""

    def fit(self, features, labels):
        self.classes_ = np.array(
            [label for label in SENTIMENT_ORDER if label in set(labels)]
        )
        self.class_to_number_ = {
            label: number for number, label in enumerate(self.classes_)
        }
        self.regressor_ = LinearRegression()
        encoded_labels = np.array([self.class_to_number_[label] for label in labels])
        self.regressor_.fit(features, encoded_labels)
        return self

    def predict(self, features):
        scores = self.regressor_.predict(features)
        class_indices = np.rint(scores).astype(int)
        class_indices = np.clip(class_indices, 0, len(self.classes_) - 1)
        return self.classes_[class_indices]


def prepare_dataset(
    dataset: pd.DataFrame,
    text_column: str = "reviews",
    label_column: str = "label",
) -> pd.DataFrame:
    required_columns = {text_column, label_column}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Kolom yang dipilih tidak ditemukan: {missing}.")
    if text_column == label_column:
        raise ValueError("Kolom teks dan kolom sentimen harus berbeda.")

    clean_data = dataset.loc[:, [text_column, label_column]].copy()
    clean_data.columns = ["reviews", "label"]
    if clean_data.isna().any().any():
        raise ValueError("Kolom reviews dan label tidak boleh berisi nilai kosong.")

    clean_data["reviews"] = clean_data["reviews"].astype(str).str.strip()
    if clean_data["reviews"].eq("").any():
        raise ValueError("Kolom reviews tidak boleh berisi teks kosong.")

    original_labels = clean_data["label"]
    numeric_labels = pd.to_numeric(original_labels, errors="coerce")
    if numeric_labels.notna().all():
        numeric_values = sorted(numeric_labels.unique())
        if len(numeric_values) == 2:
            numeric_mapping = dict(zip(numeric_values, ("Negatif", "Positif")))
        elif len(numeric_values) == 3:
            numeric_mapping = dict(zip(numeric_values, SENTIMENT_ORDER))
        else:
            raise ValueError("Dataset harus memiliki 2 atau 3 kelas sentimen.")
        clean_data["label"] = numeric_labels.map(numeric_mapping)
    else:
        normalized_labels = original_labels.astype(str).str.strip().str.casefold()
        unknown_labels = sorted(set(normalized_labels) - set(SENTIMENT_ALIASES))
        if unknown_labels:
            raise ValueError(
                "Label teks harus berupa negatif/negative, netral/neutral, "
                "atau positif/positive."
            )
        clean_data["label"] = normalized_labels.map(SENTIMENT_ALIASES)

    clean_data = clean_data.drop_duplicates().reset_index(drop=True)

    label_counts = clean_data["label"].value_counts()
    if len(label_counts) not in (2, 3):
        raise ValueError("Dataset harus memiliki 2 atau 3 kelas sentimen.")
    if label_counts.min() < 2:
        raise ValueError("Setiap kelas harus memiliki minimal 2 ulasan.")

    return clean_data


def _build_models() -> dict[str, Pipeline]:
    estimators = {
        "SVM": LinearSVC(),
        "Naive Bayes": MultinomialNB(),
        "Linear Regression": OrdinalLinearRegressionClassifier(),
    }
    return {
        name: Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=20_000,
                        sublinear_tf=True,
                    ),
                ),
                ("classifier", estimator),
            ]
        )
        for name, estimator in estimators.items()
    }


def train_and_evaluate(
    dataset: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    text_column: str = "reviews",
    label_column: str = "label",
) -> dict:
    if not 0 < test_size < 1:
        raise ValueError("Proporsi data uji harus di antara 0 dan 1.")

    clean_data = prepare_dataset(
        dataset,
        text_column=text_column,
        label_column=label_column,
    )
    test_count = ceil(len(clean_data) * test_size)
    class_count = clean_data["label"].nunique()
    if test_count < class_count or len(clean_data) - test_count < class_count:
        raise ValueError(
            "Jumlah data terlalu sedikit untuk membagi setiap kelas ke data "
            "latih dan uji. Tambahkan data atau perbesar proporsi data uji."
        )

    train_reviews, test_reviews, train_labels, test_labels = train_test_split(
        clean_data["reviews"],
        clean_data["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=clean_data["label"],
    )

    metrics = {}
    confusion_matrices = {}
    for name, model in _build_models().items():
        model.fit(train_reviews, train_labels)
        predictions = model.predict(test_reviews)
        average = "binary" if class_count == 2 else "macro"
        metrics[name] = {
            "Akurasi": accuracy_score(test_labels, predictions),
            "Precision": precision_score(
                test_labels,
                predictions,
                average=average,
                pos_label="Positif" if class_count == 2 else None,
                zero_division=0,
            ),
            "Recall": recall_score(
                test_labels,
                predictions,
                average=average,
                pos_label="Positif" if class_count == 2 else None,
                zero_division=0,
            ),
            "F1-score": f1_score(
                test_labels,
                predictions,
                average=average,
                pos_label="Positif" if class_count == 2 else None,
                zero_division=0,
            ),
        }
        confusion_matrices[name] = confusion_matrix(
            test_labels,
            predictions,
            labels=[label for label in SENTIMENT_ORDER if label in set(clean_data["label"])],
        )

    final_models = _build_models()
    for model in final_models.values():
        model.fit(clean_data["reviews"], clean_data["label"])

    first_vectorizer = next(iter(final_models.values())).named_steps["tfidf"]
    return {
        "clean_data": clean_data,
        "metrics": metrics,
        "confusion_matrices": confusion_matrices,
        "models": final_models,
        "feature_count": len(first_vectorizer.vocabulary_),
        "classes": [label for label in SENTIMENT_ORDER if label in set(clean_data["label"])],
    }
