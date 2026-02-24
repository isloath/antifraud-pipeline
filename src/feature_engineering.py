"""Aggregate feature generation for anti-fraud pipeline."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureGenerator:
    """Генерирует агрегатные фичи на основе истории транзакций пользователя.

    Для каждого временного окна (time_windows) вычисляются:
    - tx_count_<W>d      — количество транзакций за W дней
    - amount_sum_<W>d    — сумма транзакций за W дней
    - amount_mean_<W>d   — средняя сумма транзакций за W дней
    - amount_max_<W>d    — максимальная сумма транзакций за W дней
    - amount_std_<W>d    — стандартное отклонение суммы за W дней
    - amount_to_mean_ratio_<W>d — отношение текущей суммы к средней
    - unique_merchants_<W>d     — количество уникальных мерчантов (если есть)
    """

    def __init__(self, time_windows: list = None):
        self.time_windows = time_windows or [7, 30, 90]
        self.feature_cols_: list[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        user_col: str = "user_id",
        amount_col: str = "amount",
        timestamp_col: str = "timestamp",
    ) -> "FeatureGenerator":
        """Запоминает конфигурацию (fit для совместимости с Pipeline)."""
        self.user_col_ = user_col
        self.amount_col_ = amount_col
        self.timestamp_col_ = timestamp_col
        self._validate_df(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Вычисляет агрегатные фичи для каждой транзакции.

        Для каждой транзакции агрегируются только предшествующие ей события
        того же пользователя (look-back window, без утечки данных).
        """
        self._validate_df(df)
        df = df.sort_values(self.timestamp_col_).reset_index(drop=True)

        has_merchant = "merchant_id" in df.columns

        result_rows: list[dict] = []

        for idx, row in df.iterrows():
            ts: pd.Timestamp = row[self.timestamp_col_]
            user = row[self.user_col_]
            amount = row[self.amount_col_]

            user_history = df[
                (df[self.user_col_] == user) & (df[self.timestamp_col_] < ts)
            ]

            features: dict = {}

            for w in self.time_windows:
                cutoff = ts - pd.Timedelta(days=w)
                window_df = user_history[user_history[self.timestamp_col_] >= cutoff]
                amounts = window_df[self.amount_col_]

                n = len(window_df)
                features[f"tx_count_{w}d"] = n
                features[f"amount_sum_{w}d"] = amounts.sum() if n > 0 else 0.0
                features[f"amount_mean_{w}d"] = amounts.mean() if n > 0 else 0.0
                features[f"amount_max_{w}d"] = amounts.max() if n > 0 else 0.0
                features[f"amount_std_{w}d"] = amounts.std() if n > 1 else 0.0
                mean_val = features[f"amount_mean_{w}d"]
                features[f"amount_to_mean_ratio_{w}d"] = (
                    amount / mean_val if mean_val and mean_val != 0 else 1.0
                )
                if has_merchant:
                    features[f"unique_merchants_{w}d"] = (
                        window_df["merchant_id"].nunique() if n > 0 else 0
                    )

            result_rows.append(features)

        feature_df = pd.DataFrame(result_rows, index=df.index)
        self.feature_cols_ = list(feature_df.columns)
        logger.info("FeatureGenerator: создано %d фич", len(self.feature_cols_))
        return feature_df

    def fit_transform(
        self,
        df: pd.DataFrame,
        user_col: str = "user_id",
        amount_col: str = "amount",
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        return self.fit(df, user_col, amount_col, timestamp_col).transform(df)

    # ------------------------------------------------------------------
    def _validate_df(self, df: pd.DataFrame) -> None:
        required = [
            getattr(self, "user_col_", "user_id"),
            getattr(self, "amount_col_", "amount"),
            getattr(self, "timestamp_col_", "timestamp"),
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame не содержит колонок: {missing}")
