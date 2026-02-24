"""Feature binning with WoE/IV for binary classification (fraud detection)."""

import logging
import warnings

import numpy as np
import pandas as pd
from optbinning import OptimalBinning

logger = logging.getLogger(__name__)


class FeatureBinner:
    """Биннинг числовых фич с расчётом WoE и IV."""

    def __init__(self, iv_threshold: float = 0.02, max_n_bins: int = 10):
        self.iv_threshold = iv_threshold
        self.max_n_bins = max_n_bins
        self.binners_: dict[str, OptimalBinning] = {}
        self.iv_table_: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame, feature_cols: list, target_col: str):
        """Обучает OptimalBinning для каждой фичи.

        Сохраняет только фичи с IV >= iv_threshold.
        """
        self.binners_ = {}
        iv_records: list[dict] = []

        y = df[target_col].values

        for col in feature_cols:
            x = df[col].values

            ob = OptimalBinning(
                name=col,
                dtype="numerical",
                solver="cp",
                max_n_bins=self.max_n_bins,
                monotonic_trend="auto",
            )

            try:
                ob.fit(x, y)
            except Exception as exc:
                warnings.warn(
                    f"OptimalBinning не сошёлся для '{col}': {exc}",
                    stacklevel=2,
                )
                continue

            if ob.status != "OPTIMAL":
                warnings.warn(
                    f"OptimalBinning статус для '{col}': {ob.status}",
                    stacklevel=2,
                )

            table = ob.binning_table.build()
            if "Totals" not in table.index:
                continue
            iv_value = float(table.loc["Totals", "IV"])

            # Считаем только реальные бины (исключаем Special, Missing, Totals)
            real_bins = table.loc[
                ~table.index.isin(["Totals"])
                & ~table["Bin"].isin(["Special", "Missing", ""])
            ]
            n_bins = len(real_bins)

            iv_records.append(
                {
                    "feature": col,
                    "IV": iv_value,
                    "n_bins": n_bins,
                    "predictive_power": self._iv_label(iv_value),
                }
            )

            if iv_value >= self.iv_threshold:
                self.binners_[col] = ob
            else:
                logger.info(
                    "Фича '%s' исключена: IV=%.4f < порог %.4f",
                    col,
                    iv_value,
                    self.iv_threshold,
                )

        self.iv_table_ = pd.DataFrame(iv_records).sort_values(
            "IV", ascending=False, ignore_index=True
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Возвращает DataFrame с WoE-значениями для обученных фич."""
        if not self.binners_:
            raise RuntimeError("FeatureBinner не обучен — вызовите fit().")

        result = pd.DataFrame(index=df.index)
        for col, ob in self.binners_.items():
            result[col] = ob.transform(df[col].values, metric="woe")
        return result

    def get_iv_table(self) -> pd.DataFrame:
        """Возвращает таблицу с IV, количеством бинов и predictive_power.

        Колонки: feature, IV, n_bins, predictive_power.

        predictive_power:
        - "useless":  IV < 0.02
        - "weak":     0.02 <= IV < 0.1
        - "medium":   0.1  <= IV < 0.3
        - "strong":   IV >= 0.3
        """
        if self.iv_table_ is None:
            raise RuntimeError("FeatureBinner не обучен — вызовите fit().")
        return self.iv_table_

    @staticmethod
    def _iv_label(iv: float) -> str:
        if iv < 0.02:
            return "useless"
        if iv < 0.1:
            return "weak"
        if iv < 0.3:
            return "medium"
        return "strong"
