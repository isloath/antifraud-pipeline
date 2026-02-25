"""Feature selection based on pairwise correlation filtering."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Удаляет высококоррелированные признаки по верхнему треугольнику матрицы корреляций.

    При обнаружении пары с |corr| > correlation_threshold удаляется *второй*
    признак (правый в верхнем треугольнике), т.е. тот, что идёт позже по алфавиту /
    порядку столбцов.

    Parameters
    ----------
    correlation_threshold : float
        Порог абсолютной корреляции Пирсона для удаления признака (по умолчанию 0.95).
    """

    def __init__(self, correlation_threshold: float = 0.95):
        self.correlation_threshold = correlation_threshold
        self.selected_features_: list[str] | None = None
        self.removed_features_: list[str] | None = None
        self.selection_report_: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FeatureSelector":
        """Вычисляет, какие признаки нужно оставить.

        Parameters
        ----------
        df : pd.DataFrame
            Матрица признаков (только числовые столбцы).

        Returns
        -------
        self
        """
        corr_matrix = df.corr().abs()

        # Верхний треугольник (k=1 → без диагонали)
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        )

        # Колонки, у которых хотя бы одна корреляция с предыдущим признаком
        # превышает порог
        to_drop = [
            col
            for col in upper.columns
            if upper[col].max() > self.correlation_threshold
        ]

        self.selected_features_ = [c for c in df.columns if c not in to_drop]
        self.removed_features_ = to_drop

        records = []
        for col in df.columns:
            max_corr = float(upper[col].max()) if col in upper.columns else 0.0
            records.append(
                {
                    "feature": col,
                    "max_correlation": round(max_corr, 4),
                    "selected": col not in to_drop,
                }
            )
        self.selection_report_ = pd.DataFrame(records)
        # Ensure boolean dtype
        self.selection_report_["selected"] = self.selection_report_[
            "selected"
        ].astype(bool)

        logger.info(
            "FeatureSelector: отобрано %d / %d признаков (удалено: %s)",
            len(self.selected_features_),
            len(df.columns),
            to_drop,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Возвращает только отобранные признаки.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame с признаками (должны содержать все selected_features_).
        """
        if self.selected_features_ is None:
            raise RuntimeError(
                "FeatureSelector не обучен — вызовите fit()."
            )
        return df[self.selected_features_]

    def get_selection_report(self) -> pd.DataFrame:
        """Таблица со статусом каждого признака.

        Columns: feature, max_correlation, selected (bool).
        """
        if self.selection_report_ is None:
            raise RuntimeError(
                "FeatureSelector не обучен — вызовите fit()."
            )
        return self.selection_report_.copy()
