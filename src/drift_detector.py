"""Детектор дрейфа фич и предсказаний модели (PSI-based)."""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Пороги интерпретации PSI
_PSI_LOW = 0.1
_PSI_HIGH = 0.25


def _psi_label(psi: float) -> str:
    """Возвращает текстовую оценку уровня дрейфа по значению PSI."""
    if psi < _PSI_LOW:
        return "stable"
    if psi < _PSI_HIGH:
        return "moderate"
    return "high"


@dataclass
class DriftReport:
    """Отчёт о дрейфе фич и скора модели.

    Атрибуты:
        feature_psi: dict с PSI для каждой фичи.
        score_psi: PSI для распределения предсказаний (None если скор не передавался).
        drifted_features: список фич с PSI >= psi_threshold.
        drift_ratio: доля фич с дрейфом от общего числа проверенных.
        alert: флаг-строка ('NEED_RETRAIN' или 'OK').
        generated_at: метка времени генерации отчёта (UTC).
    """

    feature_psi: dict[str, float]
    score_psi: Optional[float]
    drifted_features: list[str]
    drift_ratio: float
    alert: str
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dataframe(self) -> pd.DataFrame:
        """Возвращает DataFrame с PSI по каждой фиче, отсортированный по убыванию PSI."""
        records = [
            {
                "feature": feat,
                "psi": psi,
                "drift_level": _psi_label(psi),
                "drifted": psi >= _PSI_HIGH,
            }
            for feat, psi in self.feature_psi.items()
        ]
        if self.score_psi is not None:
            records.append(
                {
                    "feature": "__score__",
                    "psi": self.score_psi,
                    "drift_level": _psi_label(self.score_psi),
                    "drifted": self.score_psi >= _PSI_HIGH,
                }
            )
        return pd.DataFrame(records).sort_values(
            "psi", ascending=False, ignore_index=True
        )


class DriftDetector:
    """Детектор дрейфа для антифрод модели на основе PSI.

    Сравнивает распределения входных фич и скора модели между эталонной
    (тренировочной) выборкой и текущими продакшн-данными.

    Алгоритм работы:
        1. fit()  — запоминает бины и гистограммы эталонных данных.
        2. detect() — вычисляет PSI для каждой фичи и для скора.
        3. generate_alert_report() — формирует итоговый DriftReport.
           Если PSI >= psi_threshold более чем для
           drift_feature_ratio_threshold доли фич → alert = 'NEED_RETRAIN'.

    Параметры:
        psi_threshold: порог PSI для классификации фичи как «задрейфовавшей».
            По умолчанию 0.25 (стандартный порог в индустрии).
        drift_feature_ratio_threshold: доля задрейфовавших фич, при превышении
            которой выставляется флаг NEED_RETRAIN. По умолчанию 0.20 (20%).
        n_bins: количество бинов при дискретизации числовых фич.
    """

    def __init__(
        self,
        psi_threshold: float = 0.25,
        drift_feature_ratio_threshold: float = 0.20,
        n_bins: int = 10,
    ):
        self.psi_threshold = psi_threshold
        self.drift_feature_ratio_threshold = drift_feature_ratio_threshold
        self.n_bins = n_bins

        # Внутреннее состояние (заполняется в fit)
        self._reference_bins: dict[str, np.ndarray] = {}
        self._reference_freqs: dict[str, np.ndarray] = {}
        self._feature_cols: list[str] = []
        self._score_col: Optional[str] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def fit(
        self,
        reference_df: pd.DataFrame,
        feature_cols: list[str],
        score_col: Optional[str] = None,
    ) -> "DriftDetector":
        """Обучает детектор на эталонных (тренировочных) данных.

        Вычисляет и сохраняет границы бинов и частоты для каждой фичи,
        а также для скора (если передан).

        Параметры:
            reference_df: DataFrame с данными на момент обучения модели.
            feature_cols: список колонок входных фич для мониторинга.
            score_col: опциональное имя колонки со скором предсказания.

        Возвращает:
            self (для цепочки вызовов).
        """
        self._feature_cols = list(feature_cols)
        self._score_col = score_col
        self._reference_bins = {}
        self._reference_freqs = {}

        cols_to_process = list(feature_cols)
        if score_col is not None:
            cols_to_process.append(score_col)

        for col in cols_to_process:
            if col not in reference_df.columns:
                warnings.warn(
                    f"Колонка '{col}' не найдена в reference_df, пропущена.",
                    stacklevel=2,
                )
                continue

            series = reference_df[col].dropna()
            if series.empty:
                warnings.warn(
                    f"Колонка '{col}' не содержит непустых значений, пропущена.",
                    stacklevel=2,
                )
                continue

            if pd.api.types.is_numeric_dtype(series):
                bins, freqs = self._build_numeric_bins(series.values)
            else:
                bins, freqs = self._build_categorical_bins(series)

            self._reference_bins[col] = bins
            self._reference_freqs[col] = freqs

        self._fitted = True
        logger.info(
            "DriftDetector обучен: %d фич, score_col=%s",
            len(self._feature_cols),
            score_col,
        )
        return self

    def detect(self, current_df: pd.DataFrame) -> dict[str, float]:
        """Вычисляет PSI для каждой фичи и скора по текущим данным.

        Параметры:
            current_df: DataFrame с текущими (продакшн) данными.

        Возвращает:
            Словарь {имя_колонки: psi_value}. Значение NaN означает,
            что PSI для данной колонки вычислить не удалось.
        """
        self._check_fitted()

        cols_to_check = list(self._feature_cols)
        if self._score_col is not None:
            cols_to_check.append(self._score_col)

        psi_results: dict[str, float] = {}

        for col in cols_to_check:
            if col not in self._reference_bins:
                continue

            if col not in current_df.columns:
                warnings.warn(
                    f"Колонка '{col}' не найдена в current_df, PSI=NaN.",
                    stacklevel=2,
                )
                psi_results[col] = float("nan")
                continue

            series = current_df[col].dropna()
            if series.empty:
                warnings.warn(
                    f"Колонка '{col}' пуста в current_df, PSI=NaN.",
                    stacklevel=2,
                )
                psi_results[col] = float("nan")
                continue

            ref_bins = self._reference_bins[col]
            ref_freqs = self._reference_freqs[col]

            if pd.api.types.is_numeric_dtype(series):
                cur_freqs = self._get_numeric_freqs(series.values, ref_bins)
            else:
                cur_freqs = self._get_categorical_freqs(series, ref_bins)

            psi_results[col] = self._compute_psi(ref_freqs, cur_freqs)

        return psi_results

    def generate_alert_report(self, current_df: pd.DataFrame) -> DriftReport:
        """Формирует итоговый отчёт о дрейфе и Alert-флаг.

        Правило выставления флага:
            Если PSI >= psi_threshold у доли фич > drift_feature_ratio_threshold
            → alert = 'NEED_RETRAIN', иначе → alert = 'OK'.

        Параметры:
            current_df: DataFrame с текущими (продакшн) данными.

        Возвращает:
            DriftReport с PSI по каждой фиче, score_psi, списком
            задрейфовавших фич, долей дрейфа и alert-флагом.
        """
        self._check_fitted()

        all_psi = self.detect(current_df)

        # Разделяем PSI фич и скора
        feature_psi = {
            col: psi for col, psi in all_psi.items() if col != self._score_col
        }
        score_psi = (
            all_psi.get(self._score_col)
            if self._score_col is not None
            else None
        )

        # Фичи с дрейфом (исключаем NaN)
        drifted = [
            col
            for col, psi in feature_psi.items()
            if not np.isnan(psi) and psi >= self.psi_threshold
        ]

        valid_count = sum(
            1 for psi in feature_psi.values() if not np.isnan(psi)
        )
        drift_ratio = len(drifted) / valid_count if valid_count > 0 else 0.0

        alert = (
            "NEED_RETRAIN"
            if drift_ratio > self.drift_feature_ratio_threshold
            else "OK"
        )

        logger.info(
            "Drift report: %d/%d фич задрейфовали (%.1f%%), alert=%s",
            len(drifted),
            valid_count,
            drift_ratio * 100,
            alert,
        )

        return DriftReport(
            feature_psi=feature_psi,
            score_psi=score_psi,
            drifted_features=drifted,
            drift_ratio=drift_ratio,
            alert=alert,
        )

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _build_numeric_bins(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Строит квантильные бины и частоты для числовой эталонной колонки."""
        # Квантильное биннирование равномерно заполняет каждый бин
        edges = np.percentile(values, np.linspace(0, 100, self.n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            # Вырожденный случай: все значения одинаковы
            edges = np.array([values.min() - 1e-9, values.max() + 1e-9])

        counts, _ = np.histogram(values, bins=edges)
        freqs = counts / counts.sum()
        return edges, freqs

    def _build_categorical_bins(
        self, series: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Строит список категорий и частоты для категориальной эталонной колонки."""
        value_counts = series.value_counts(normalize=True)
        categories = value_counts.index.to_numpy()
        freqs = value_counts.values
        return categories, freqs

    def _get_numeric_freqs(
        self, values: np.ndarray, bins: np.ndarray
    ) -> np.ndarray:
        """Вычисляет частоты текущих числовых данных по заранее заданным бинам."""
        counts, _ = np.histogram(values, bins=bins)
        total = counts.sum()
        return counts / total if total > 0 else counts.astype(float)

    def _get_categorical_freqs(
        self, series: pd.Series, categories: np.ndarray
    ) -> np.ndarray:
        """Вычисляет частоты текущих данных по эталонному списку категорий."""
        total = len(series)
        freqs = np.array(
            [
                (series == cat).sum() / total if total > 0 else 0.0
                for cat in categories
            ]
        )
        return freqs

    @staticmethod
    def _compute_psi(
        expected_freqs: np.ndarray, actual_freqs: np.ndarray
    ) -> float:
        """Вычисляет Population Stability Index (PSI).

        Формула: PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)

        Интерпретация:
            PSI < 0.10  — распределение стабильно (stable)
            PSI < 0.25  — умеренный дрейф (moderate), требует мониторинга
            PSI >= 0.25 — значительный дрейф (high), модель требует переобучения

        Параметры:
            expected_freqs: частоты эталонного распределения (сумма ≈ 1).
            actual_freqs: частоты текущего распределения (сумма ≈ 1).

        Возвращает:
            Значение PSI (≥ 0).
        """
        eps = 1e-6  # защита от log(0) и деления на 0
        expected = np.clip(expected_freqs, eps, None)
        actual = np.clip(actual_freqs, eps, None)

        # Нормализуем на случай, если суммы немного отличаются от 1
        expected = expected / expected.sum()
        actual = actual / actual.sum()

        psi = np.sum((actual - expected) * np.log(actual / expected))
        return float(psi)

    def _check_fitted(self) -> None:
        """Проверяет, что детектор обучен."""
        if not self._fitted:
            raise RuntimeError("DriftDetector не обучен — вызовите fit().")
