"""Feature engineering for anti-fraud transaction data."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureGenerator:
    """Генерирует признаки из сырых транзакционных данных.

    Produces 13 features:
        log_amount, amount_zscore,
        hour_of_day, day_of_week, is_weekend,
        card_tx_count, card_avg_amount, card_std_amount, card_max_amount,
        card_amount_ratio,
        merchant_tx_count, merchant_avg_amount, merchant_fraud_proxy.
    """

    EXPECTED_BASE_FEATURES = [
        "log_amount",
        "amount_zscore",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "card_tx_count",
        "card_avg_amount",
        "card_std_amount",
        "card_max_amount",
        "card_amount_ratio",
        "merchant_tx_count",
        "merchant_avg_amount",
        "merchant_fraud_proxy",
    ]

    def __init__(
        self,
        timestamp_col: str = "timestamp",
        card_col: str = "card_id",
        merchant_col: str = "merchant_id",
        amount_col: str = "amount",
    ):
        self.timestamp_col = timestamp_col
        self.card_col = card_col
        self.merchant_col = merchant_col
        self.amount_col = amount_col

        self.feature_names_: list[str] | None = None
        self._card_stats: pd.DataFrame | None = None
        self._merchant_stats: pd.DataFrame | None = None
        self._global_mean: float | None = None
        self._global_std: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обучается на df и возвращает матрицу признаков."""
        self._fit(df)
        return self._generate(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применяет обученные статистики к новым данным."""
        if self._card_stats is None:
            raise RuntimeError(
                "FeatureGenerator не обучен — вызовите fit_transform()."
            )
        return self._generate(df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit(self, df: pd.DataFrame) -> None:
        """Вычисляет агрегированные статистики на тренировочных данных."""
        amt = df[self.amount_col]
        self._global_mean = float(amt.mean())
        self._global_std = float(amt.std()) or 1.0

        self._card_stats = (
            df.groupby(self.card_col)[self.amount_col]
            .agg(["count", "mean", "std", "max"])
            .rename(
                columns={
                    "count": "card_tx_count",
                    "mean": "card_avg_amount",
                    "std": "card_std_amount",
                    "max": "card_max_amount",
                }
            )
        )
        # Single-transaction cards have std=NaN → fill with 0
        self._card_stats["card_std_amount"] = self._card_stats[
            "card_std_amount"
        ].fillna(0.0)

        self._merchant_stats = (
            df.groupby(self.merchant_col)[self.amount_col]
            .agg(["count", "mean"])
            .rename(
                columns={
                    "count": "merchant_tx_count",
                    "mean": "merchant_avg_amount",
                }
            )
        )

    def _generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Строит матрицу признаков для переданного DataFrame."""
        res = pd.DataFrame(index=df.index)
        amt = df[self.amount_col]

        # --- Amount features ---
        res["log_amount"] = np.log1p(amt)
        res["amount_zscore"] = (amt - self._global_mean) / self._global_std

        # --- Time features ---
        ts = pd.to_datetime(df[self.timestamp_col])
        res["hour_of_day"] = ts.dt.hour
        res["day_of_week"] = ts.dt.dayofweek
        res["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

        # --- Card aggregation features ---
        card_merged = df[[self.card_col]].join(
            self._card_stats, on=self.card_col
        )
        res["card_tx_count"] = card_merged["card_tx_count"].fillna(1)
        res["card_avg_amount"] = card_merged["card_avg_amount"].fillna(
            self._global_mean
        )
        res["card_std_amount"] = card_merged["card_std_amount"].fillna(0.0)
        res["card_max_amount"] = card_merged["card_max_amount"].fillna(amt)
        card_avg_filled = card_merged["card_avg_amount"].fillna(
            self._global_mean
        )
        res["card_amount_ratio"] = amt / (card_avg_filled + 1e-9)

        # --- Merchant aggregation features ---
        merch_merged = df[[self.merchant_col]].join(
            self._merchant_stats, on=self.merchant_col
        )
        res["merchant_tx_count"] = merch_merged["merchant_tx_count"].fillna(1)
        res["merchant_avg_amount"] = merch_merged["merchant_avg_amount"].fillna(
            self._global_mean
        )
        merch_avg_filled = merch_merged["merchant_avg_amount"].fillna(
            self._global_mean
        )
        res["merchant_fraud_proxy"] = amt / (merch_avg_filled + 1e-9)

        self.feature_names_ = list(res.columns)
        logger.debug("FeatureGenerator: сгенерировано %d признаков", len(self.feature_names_))
        return res
