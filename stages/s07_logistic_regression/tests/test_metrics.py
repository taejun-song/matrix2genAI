from __future__ import annotations

import numpy as np

from stages.s07_logistic_regression.starter.metrics import (
    accuracy,
    classification_report,
    confusion_matrix,
    precision_recall_f1,
    roc_auc_score,
)


class TestAccuracy:
    def test_perfect_predictions(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])

        acc = accuracy(y_true, y_pred)

        assert acc == 1.0

    def test_worst_predictions(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1])

        acc = accuracy(y_true, y_pred)

        assert acc == 0.0

    def test_partial_correct(self):
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0])

        acc = accuracy(y_true, y_pred)

        assert acc == 0.8

    def test_multiclass(self):
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])

        acc = accuracy(y_true, y_pred)

        assert acc == 0.8


class TestConfusionMatrix:
    def test_binary_perfect(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])

        cm = confusion_matrix(y_true, y_pred, n_classes=2)

        expected = np.array([[2, 0], [0, 2]])
        np.testing.assert_array_equal(cm, expected)

    def test_binary_with_errors(self):
        y_true = np.array([1, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1, 0, 1])

        cm = confusion_matrix(y_true, y_pred, n_classes=2)

        expected = np.array([[2, 1], [1, 2]])
        np.testing.assert_array_equal(cm, expected)

    def test_multiclass(self):
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])

        cm = confusion_matrix(y_true, y_pred, n_classes=3)

        expected = np.array([[2, 0, 0], [0, 2, 0], [0, 1, 0]])
        np.testing.assert_array_equal(cm, expected)

    def test_auto_detect_classes(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])

        cm = confusion_matrix(y_true, y_pred)

        assert cm.shape == (3, 3)

    def test_three_classes_complex(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 2, 0])

        cm = confusion_matrix(y_true, y_pred, n_classes=3)

        expected = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        np.testing.assert_array_equal(cm, expected)


class TestPrecisionRecallF1:
    def test_binary_perfect(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])

        precision, recall, f1 = precision_recall_f1(y_true, y_pred, average="binary")

        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_binary_with_errors(self):
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0])

        precision, recall, f1 = precision_recall_f1(y_true, y_pred, average="binary")

        assert precision == 1.0
        assert recall == 2 / 3
        np.testing.assert_allclose(f1, 0.8, rtol=1e-5)

    def test_binary_all_negative(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        precision, recall, f1 = precision_recall_f1(y_true, y_pred, average="binary")

        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_binary_false_positives(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])

        precision, recall, f1 = precision_recall_f1(y_true, y_pred, average="binary")

        assert precision == 0.5
        assert recall == 1.0
        np.testing.assert_allclose(f1, 2 / 3, rtol=1e-5)

    def test_binary_false_negatives(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        precision, recall, f1 = precision_recall_f1(y_true, y_pred, average="binary")

        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_multiclass_macro(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        precision, recall, f1 = precision_recall_f1(y_true, y_pred, average="macro")

        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0


class TestROCAUCScore:
    def test_perfect_ranking(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred_proba = np.array([0.9, 0.3, 0.8, 0.2])

        auc = roc_auc_score(y_true, y_pred_proba)

        assert auc == 1.0

    def test_worst_ranking(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])

        auc = roc_auc_score(y_true, y_pred_proba)

        assert auc == 0.0

    def test_random_ranking(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred_proba = np.array([0.5, 0.5, 0.5, 0.5])

        auc = roc_auc_score(y_true, y_pred_proba)

        assert auc == 0.5

    def test_good_ranking(self):
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred_proba = np.array([0.9, 0.8, 0.4, 0.3, 0.7, 0.2])

        auc = roc_auc_score(y_true, y_pred_proba)

        assert auc > 0.8


class TestClassificationReport:
    def test_binary_report(self):
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0])
        y_pred_proba = np.array([0.9, 0.1, 0.4, 0.8, 0.2])

        report = classification_report(y_true, y_pred, y_pred_proba)

        assert "accuracy" in report
        assert "precision" in report
        assert "recall" in report
        assert "f1" in report
        assert "roc_auc" in report
        assert "confusion_matrix" in report

        assert report["accuracy"] == 0.8
        assert 0 <= report["roc_auc"] <= 1

    def test_multiclass_report(self):
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])

        report = classification_report(y_true, y_pred)

        assert "accuracy" in report
        assert "precision" in report
        assert "recall" in report
        assert "f1" in report
        assert "confusion_matrix" in report

        assert report["accuracy"] == 0.8

    def test_perfect_predictions(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        y_pred_proba = np.array([1.0, 0.0, 1.0, 0.0])

        report = classification_report(y_true, y_pred, y_pred_proba)

        assert report["accuracy"] == 1.0
        assert report["precision"] == 1.0
        assert report["recall"] == 1.0
        assert report["f1"] == 1.0

    def test_without_probabilities(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0])

        report = classification_report(y_true, y_pred)

        assert "accuracy" in report
        assert "precision" in report
        assert "recall" in report
        assert "f1" in report
