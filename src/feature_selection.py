"""Feature selection by IV threshold and correlation filtering."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Отбор фич по IV-порогу и матрице корреляций.

    Алгоритм:
    1. Из IV-таблицы (binning) берём фичи с IV >= iv_threshold.
    2. Среди оставшихся строим матрицу корреляций Пирсона.
    3. Для каждой пары (i, j) с |corr| > correlation_threshold исключаем
       фичу с меньшим IV.
    """

    def __init__(
        self,
        iv_threshold: float = 0.1,
        correlation_threshold: float = 0.8,
    ):
        self.iv_threshold = iv_threshold
        self.correlation_threshold = correlation_threshold
        self.selected_features_: list[str] = []
        self.iv_table_: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        iv_table: pd.DataFrame,
    ) -> "FeatureSelector":
        """Определяет список отобранных фич.

        Parameters
        ----------
        df:
            DataFrame с уже забиненными WoE-фичами.
        iv_table:
            Таблица вида (feature, IV, ...) — выход из FeatureBinner.get_iv_table().
        """
        self.iv_table_ = iv_table.copy()

        # 1. Фильтр по IV
        iv_pass = iv_table.loc[
            iv_table["IV"] >= self.iv_threshold, "feature"
        ].tolist()

        # Оставляем только те, что есть в df
        iv_pass = [f for f in iv_pass if f in df.columns]

        if not iv_pass:
            logger.warning(
                "FeatureSelector: ни одна фича не прошла IV-фильтр (порог=%.2f)",
                self.iv_threshold,
            )
            self.selected_features_ = []
            return self

        # 2. Фильтр по корреляции (greedy: удаляем фичу с меньшим IV)
        iv_map: dict[str, float] = dict(
            zip(iv_table["feature"], iv_table["IV"])
        )
        iv_pass_sorted = sorted(iv_pass, key=lambda f: iv_map.get(f, 0), reverse=True)

        corr_matrix = df[iv_pass_sorted].corr().abs()
        excluded: set[str] = set()

        for i, feat_i in enumerate(iv_pass_sorted):
            if feat_i in excluded:
                continue
            for feat_j in iv_pass_sorted[i + 1 :]:
                if feat_j in excluded:
                    continue
                if corr_matrix.loc[feat_i, feat_j] > self.correlation_threshold:
                    # feat_j имеет меньший IV (список отсортирован по убыванию IV)
                    excluded.add(feat_j)
                    logger.info(
                        "Исключена '%s' (corr=%.3f с '%s', IV=%.4f < %.4f)",
                        feat_j,
                        corr_matrix.loc[feat_i, feat_j],
                        feat_i,
                        iv_map.get(feat_j, 0),
                        iv_map.get(feat_i, 0),
                    )

        self.selected_features_ = [f for f in iv_pass_sorted if f not in excluded]
        logger.info(
            "FeatureSelector: отобрано %d фич из %d (IV-фильтр), %d исключены по корреляции",
            len(self.selected_features_),
            len(iv_pass),
            len(excluded),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Возвращает DataFrame только с отобранными фичами."""
        if not self.selected_features_:
            raise RuntimeError("FeatureSelector не обучен — вызовите fit().")
        available = [f for f in self.selected_features_ if f in df.columns]
        return df[available].copy()

    def get_selected_features(self) -> list[str]:
        return list(self.selected_features_)
