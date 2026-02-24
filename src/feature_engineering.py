"""Feature engineering for antifraud transaction data.

Generates time-window aggregates, ratio, and velocity features
without look-ahead bias (current transaction is excluded from all
historical aggregations via closed='left' rolling windows).
"""

import numpy as np
import pandas as pd


class FeatureGenerator:
    """Генерация агрегатных фич для антифрод модели."""

    def __init__(self, time_windows: list = [7, 30, 90]):
        """
        Args:
            time_windows: список дней для агрегации (по умолчанию 7, 30, 90).
        """
        self.time_windows = time_windows
        self.feature_names_: list[str] = []

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def generate_features(
        self,
        df: pd.DataFrame,
        user_col: str = "user_id",
        amount_col: str = "amount",
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Генерирует агрегатные фичи без look-ahead bias.

        Look-ahead bias предотвращается через rolling с closed='left':
        окно определяется как [t - window, t) — текущая транзакция
        НЕ входит в расчёт агрегатов.

        NaN заполняются нулями: интерпретируется как «нет истории».
        (Для count и sum это семантически корректно; для mean/std/max
        нуль означает отсутствие опорных данных и безопасен при inference.)

        Args:
            df:            DataFrame с транзакциями.
            user_col:      колонка с ID пользователя.
            amount_col:    колонка с суммой транзакции.
            timestamp_col: колонка с временем транзакции.

        Returns:
            Копия df с добавленными колонками фич.
        """
        df = df.copy()

        # Приводим временную метку к UTC; локальные и naive-метки обрабатываются
        # единообразно — ошибки при смешанных временных зонах исключены.
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

        # Сортировка обязательна: rolling по времени требует монотонного индекса.
        df = df.sort_values([user_col, timestamp_col]).reset_index(drop=True)

        feature_names: list[str] = []

        # ------------------------------------------------------------------ #
        # 1. Агрегаты за временные окна                                        #
        # ------------------------------------------------------------------ #
        # closed='left' задаёт полуоткрытый интервал [t-window, t):
        #   — строки до текущей транзакции входят в окно,
        #   — текущая транзакция НЕ входит → нет look-ahead bias.
        # min_periods=0 гарантирует sum/count=0 при пустом окне
        # вместо NaN; mean/std/max всё равно дают NaN при count=0
        # и заполняются ниже.

        def _rolling_agg(group: pd.DataFrame, window_str: str) -> pd.DataFrame:
            """Возвращает rolling-агрегаты для одного пользователя."""
            # Индекс серии — datetime; rolling работает по нему.
            series = pd.Series(
                group[amount_col].values,
                index=group[timestamp_col].values,
                dtype=float,
            )
            roll = series.rolling(window=window_str, closed="left", min_periods=0)
            return pd.DataFrame(
                {
                    "sum": roll.sum().to_numpy(),
                    "mean": roll.mean().to_numpy(),
                    "std": roll.std().to_numpy(),
                    "count": roll.count().to_numpy(),
                    "max": roll.max().to_numpy(),
                },
                index=group.index,
            )

        for window in self.time_windows:
            window_str = f"{window}D"
            agg = df.groupby(user_col, group_keys=False).apply(
                lambda g, w=window_str: _rolling_agg(g, w)
            )

            df[f"{amount_col}_sum_{window}d"] = agg["sum"]
            df[f"{amount_col}_mean_{window}d"] = agg["mean"]
            df[f"{amount_col}_std_{window}d"] = agg["std"]
            df[f"{amount_col}_count_{window}d"] = agg["count"]
            df[f"{amount_col}_max_{window}d"] = agg["max"]

            feature_names.extend(
                [
                    f"{amount_col}_sum_{window}d",
                    f"{amount_col}_mean_{window}d",
                    f"{amount_col}_std_{window}d",
                    f"{amount_col}_count_{window}d",
                    f"{amount_col}_max_{window}d",
                ]
            )

        # ------------------------------------------------------------------ #
        # 2. Ratios: текущая сумма / исторический агрегат                      #
        # ------------------------------------------------------------------ #
        # Деление на ноль (нет истории) заменяется нулём.

        for window in self.time_windows:
            mean_col = f"{amount_col}_mean_{window}d"
            max_col = f"{amount_col}_max_{window}d"

            df[f"{amount_col}_to_mean_{window}d"] = np.where(
                df[mean_col] != 0,
                df[amount_col] / df[mean_col],
                0.0,
            )
            df[f"{amount_col}_to_max_{window}d"] = np.where(
                df[max_col] != 0,
                df[amount_col] / df[max_col],
                0.0,
            )

            feature_names.extend(
                [
                    f"{amount_col}_to_mean_{window}d",
                    f"{amount_col}_to_max_{window}d",
                ]
            )

        # ------------------------------------------------------------------ #
        # 3. Velocity фичи                                                     #
        # ------------------------------------------------------------------ #

        def _velocity_agg(group: pd.DataFrame) -> pd.DataFrame:
            """Velocity-фичи для одного пользователя без look-ahead bias."""
            # Группа уже отсортирована по времени (df отсортирован выше).
            ts = group[timestamp_col]

            # Dummy-серия с datetime-индексом: 1 за каждую транзакцию.
            # rolling по datetime-индексу с closed='left' исключает текущую строку.
            indicator = pd.Series(1.0, index=ts.values)

            count_1h = indicator.rolling("1h", closed="left", min_periods=0).sum()
            count_24h = indicator.rolling("24h", closed="left", min_periods=0).sum()

            # diff() от отсортированного ряда = время с предыдущей транзакции.
            # Для первой транзакции → NaN → заполним 0 ниже (нет предыдущей).
            hours_since = ts.diff().dt.total_seconds() / 3600.0

            return pd.DataFrame(
                {
                    "txn_count_last_1h": count_1h.to_numpy(),
                    "txn_count_last_24h": count_24h.to_numpy(),
                    "hours_since_last_txn": hours_since.to_numpy(),
                },
                index=group.index,
            )

        velocity = df.groupby(user_col, group_keys=False).apply(_velocity_agg)
        df["txn_count_last_1h"] = velocity["txn_count_last_1h"]
        df["txn_count_last_24h"] = velocity["txn_count_last_24h"]
        df["hours_since_last_txn"] = velocity["hours_since_last_txn"]

        feature_names.extend(
            ["txn_count_last_1h", "txn_count_last_24h", "hours_since_last_txn"]
        )

        # ------------------------------------------------------------------ #
        # Заполнение NaN нулями                                                #
        # ------------------------------------------------------------------ #
        df[feature_names] = df[feature_names].fillna(0.0)

        self.feature_names_ = feature_names
        return df

    def get_feature_names(self) -> list:
        """Возвращает список сгенерированных фич."""
        return self.feature_names_


# --------------------------------------------------------------------------- #
# Тест и верификация look-ahead bias                                            #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=" * 60)
    print("FeatureGenerator: smoke-test on synthetic data")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Генерация тестового датасета
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(42)
    n_users = 50
    n_txns = 500

    user_ids = [f"user_{i:03d}" for i in range(n_users)]
    base = pd.Timestamp("2023-01-01", tz="UTC")

    test_df = pd.DataFrame(
        {
            "user_id": rng.choice(user_ids, size=n_txns),
            "amount": np.round(rng.exponential(scale=500, size=n_txns) + 10, 2),
            "timestamp": [
                base + pd.Timedelta(seconds=float(s))
                for s in rng.uniform(0, 90 * 86400, size=n_txns)
            ],
            "merchant_category": rng.choice(
                ["retail", "food", "travel", "entertainment", "healthcare"],
                size=n_txns,
            ),
            "is_fraud": rng.choice([0, 1], size=n_txns, p=[0.95, 0.05]),
        }
    )

    fg = FeatureGenerator(time_windows=[7, 30, 90])
    result = fg.generate_features(test_df)

    print("\n--- df.head() ---")
    display_cols = (
        ["user_id", "amount", "timestamp"]
        + [c for c in fg.get_feature_names() if "7d" in c][:6]
    )
    print(result[display_cols].head().to_string(index=False))

    print("\n--- Сгенерированные фичи ---")
    for name in fg.get_feature_names():
        print(" ", name)
    print(f"\nВсего фич: {len(fg.get_feature_names())}")

    # -----------------------------------------------------------------------
    # Верификация: нет look-ahead bias в amount_mean_7d
    # -----------------------------------------------------------------------
    print("\n--- Верификация look-ahead bias для amount_mean_7d ---")

    # Берём пользователя с наибольшим числом транзакций
    active_user = result.groupby("user_id").size().idxmax()
    user_txns = result[result["user_id"] == active_user].sort_values("timestamp")

    errors = 0
    for _, row in user_txns.iterrows():
        t = row["timestamp"]
        computed = row["amount_mean_7d"]

        # Ручной расчёт: предшествующие транзакции за 7 дней (без текущей)
        prev = user_txns[
            (user_txns["timestamp"] < t)
            & (user_txns["timestamp"] >= t - pd.Timedelta(days=7))
        ]

        if len(prev) > 0:
            expected = prev["amount"].mean()
            if abs(computed - expected) > 1e-6:
                print(
                    f"  FAIL t={t}: computed={computed:.4f}, expected={expected:.4f}"
                )
                errors += 1
        else:
            if computed != 0.0:
                print(f"  FAIL t={t}: expected 0 (no history), got {computed:.4f}")
                errors += 1

    if errors == 0:
        print(
            f"  OK — проверено {len(user_txns)} транзакций "
            f"для '{active_user}', look-ahead bias не обнаружен."
        )
    else:
        print(f"  FAIL — обнаружено {errors} нарушений!")

    print("\nNaN-значений в фичах:", result[fg.get_feature_names()].isna().sum().sum())
