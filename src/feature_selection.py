"""Feature selection by IV, PSI, and correlation analysis for antifraud models."""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from optbinning import OptimalBinning

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Отбор фич на основе IV, PSI и корреляционного анализа.

    Этапы отбора в fit():
    1. Удаление фич с IV < iv_threshold (нет предиктивной силы).
    2. Удаление фич с PSI > psi_threshold (нестабильны между train и OOT).
    3. Удаление высококоррелированных фич (оставляем ту, у которой IV выше).
    """

    def __init__(
        self,
        iv_threshold: float = 0.1,
        correlation_threshold: float = 0.8,
        psi_threshold: float = 0.2,
    ):
        self.iv_threshold = iv_threshold
        self.correlation_threshold = correlation_threshold
        self.psi_threshold = psi_threshold
        self.selected_features_: list = []
        self.dropped_features_: dict = {}   # feature -> reason
        self._iv_values_: dict = {}
        self._psi_values_: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_psi(
        expected: np.ndarray,
        actual: np.ndarray,
        buckets: int = 10,
    ) -> float:
        """Считает PSI (Population Stability Index) между двумя распределениями.

        Parameters
        ----------
        expected : array-like
            Базовое (train) распределение.
        actual : array-like
            Новое (OOT) распределение.
        buckets : int
            Число бакетов; границы строятся по квантилям expected.

        Returns
        -------
        float
            PSI < 0.1 — стабильно, 0.1–0.2 — небольшие изменения, > 0.2 — нестабильно.
        """
        expected = np.asarray(expected, dtype=float)
        actual = np.asarray(actual, dtype=float)

        # Границы бакетов по квантилям expected (убираем дубликаты)
        breakpoints = np.nanpercentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints = np.unique(breakpoints)

        expected_counts = np.histogram(expected, bins=breakpoints)[0].astype(float)
        actual_counts = np.histogram(actual, bins=breakpoints)[0].astype(float)

        # Защита от нулевых бакетов — добавляем сглаживание 1e-6
        eps = 1e-6
        n_buckets = len(expected_counts)
        expected_pct = (expected_counts + eps) / (expected_counts.sum() + eps * n_buckets)
        actual_pct = (actual_counts + eps) / (actual_counts.sum() + eps * n_buckets)

        psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
        return psi

    def fit(
        self,
        df_train: pd.DataFrame,
        feature_cols: list,
        target_col: str,
        df_oot: Optional[pd.DataFrame] = None,
    ) -> "FeatureSelector":
        """Отбирает фичи в три этапа.

        Parameters
        ----------
        df_train : pd.DataFrame
            Обучающая выборка (содержит feature_cols и target_col).
        feature_cols : list[str]
            Список кандидатов для отбора.
        target_col : str
            Имя бинарной целевой переменной (0/1).
        df_oot : pd.DataFrame, optional
            Out-of-Time выборка для расчёта PSI. Если None — этап 2 пропускается.

        Returns
        -------
        self
        """
        self.selected_features_ = []
        self.dropped_features_ = {}
        self._iv_values_ = {}
        self._psi_values_ = {}

        y_train = df_train[target_col].values

        # ---- Этап 1: IV-фильтр ----------------------------------------
        logger.info("Этап 1: расчёт IV для %d фич...", len(feature_cols))
        iv_passed: list = []

        for col in feature_cols:
            iv = self._calculate_iv(df_train[col].values, y_train, col)
            self._iv_values_[col] = iv

            if iv < self.iv_threshold:
                self.dropped_features_[col] = "low_iv"
                logger.info(
                    "Дроп '%s': IV=%.4f < порог %.4f", col, iv, self.iv_threshold
                )
            else:
                iv_passed.append(col)

        logger.info(
            "После IV-фильтра: %d/%d фич прошло.", len(iv_passed), len(feature_cols)
        )

        # ---- Этап 2: PSI-фильтр ---------------------------------------
        if df_oot is not None:
            logger.info("Этап 2: расчёт PSI для %d фич...", len(iv_passed))
            psi_passed: list = []

            for col in iv_passed:
                psi = self.calculate_psi(df_train[col].values, df_oot[col].values)
                self._psi_values_[col] = psi

                if psi > self.psi_threshold:
                    self.dropped_features_[col] = "high_psi"
                    logger.info(
                        "Дроп '%s': PSI=%.4f > порог %.4f",
                        col,
                        psi,
                        self.psi_threshold,
                    )
                else:
                    psi_passed.append(col)

            logger.info(
                "После PSI-фильтра: %d/%d фич прошло.", len(psi_passed), len(iv_passed)
            )
        else:
            psi_passed = iv_passed
            logger.info("Этап 2 пропущен: df_oot не передан.")

        # ---- Этап 3: Корреляционный фильтр ----------------------------
        logger.info("Этап 3: корреляционный анализ для %d фич...", len(psi_passed))
        self.selected_features_ = self._drop_correlated(df_train, psi_passed)
        logger.info("Итого отобрано фич: %d", len(self.selected_features_))

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Возвращает DataFrame только с отобранными фичами."""
        if not self.selected_features_:
            raise RuntimeError("FeatureSelector не обучен — вызовите fit().")
        return df[self.selected_features_].copy()

    def get_selection_report(self) -> pd.DataFrame:
        """Возвращает отчёт об отборе фич.

        Колонки
        -------
        feature      : имя фичи
        IV           : значение Information Value
        psi_value    : значение PSI (NaN если df_oot не передавался)
        selected     : True если фича отобрана
        drop_reason  : "low_iv" | "high_psi" | "high_correlation_with_{name}" | None
        """
        if not self._iv_values_:
            raise RuntimeError("FeatureSelector не обучен — вызовите fit().")

        records = [
            {
                "feature": feat,
                "IV": self._iv_values_.get(feat, np.nan),
                "psi_value": self._psi_values_.get(feat, np.nan),
                "selected": feat in self.selected_features_,
                "drop_reason": self.dropped_features_.get(feat, None),
            }
            for feat in self._iv_values_
        ]

        return (
            pd.DataFrame(records)
            .sort_values("IV", ascending=False, ignore_index=True)
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _calculate_iv(self, x: np.ndarray, y: np.ndarray, name: str) -> float:
        """Рассчитывает IV через OptimalBinning."""
        ob = OptimalBinning(
            name=name,
            dtype="numerical",
            solver="cp",
            max_n_bins=10,
            monotonic_trend="auto",
        )
        try:
            ob.fit(x, y)
        except Exception as exc:
            warnings.warn(
                f"OptimalBinning не сошёлся для '{name}': {exc}", stacklevel=2
            )
            return 0.0

        table = ob.binning_table.build()
        if "Totals" not in table.index:
            return 0.0
        return float(table.loc["Totals", "IV"])

    def _drop_correlated(self, df: pd.DataFrame, features: list) -> list:
        """Удаляет высококоррелированные фичи, оставляя ту, у которой IV выше."""
        if len(features) < 2:
            return features

        corr_matrix = df[features].corr().abs()
        to_drop: set = set()

        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                fi, fj = features[i], features[j]
                if fi in to_drop or fj in to_drop:
                    continue
                if corr_matrix.loc[fi, fj] > self.correlation_threshold:
                    iv_i = self._iv_values_.get(fi, 0.0)
                    iv_j = self._iv_values_.get(fj, 0.0)
                    loser, winner = (fj, fi) if iv_i >= iv_j else (fi, fj)
                    to_drop.add(loser)
                    self.dropped_features_[loser] = f"high_correlation_with_{winner}"
                    logger.info(
                        "Дроп '%s' (corr=%.4f с '%s', IV %.4f < %.4f)",
                        loser,
                        corr_matrix.loc[fi, fj],
                        winner,
                        self._iv_values_.get(loser, 0.0),
                        self._iv_values_.get(winner, 0.0),
                    )

        return [f for f in features if f not in to_drop]


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stdout,
    )

    rng = np.random.default_rng(42)
    n_train, n_oot = 5_000, 2_000

    # --- Train ----------------------------------------------------------------
    fraud_train = rng.integers(0, 2, n_train)  # бинарный таргет

    df_train = pd.DataFrame(
        {
            # Сильные предикторы
            "amount":      rng.normal(500, 200, n_train) + fraud_train * 300,
            "velocity_1h": rng.exponential(2, n_train) + fraud_train * 5,
            "card_age":    rng.uniform(0, 10, n_train) - fraud_train * 2,
            # Слабый предиктор (IV ожидается ниже порога)
            "weak_feat":   rng.normal(0, 1, n_train) + fraud_train * 0.05,
            # Высококоррелированная с amount (должна вылететь на этапе 3)
            "amount_copy": None,
            # Полностью некоррелированный шум
            "noise_1":     rng.normal(0, 1, n_train),
            "noise_2":     rng.uniform(0, 1, n_train),
            # Нестабильная фича: хороший IV на трейне, но сильный дрейф в OOT (PSI > 0.2)
            "drifting":    rng.normal(0, 1, n_train) + fraud_train * 3,
            # Ещё два умеренных предиктора
            "dist_center": rng.exponential(10, n_train) + fraud_train * 8,
            "hour_sin":    np.sin(rng.uniform(0, 2 * np.pi, n_train)) + fraud_train * 0.3,
        }
    )
    # amount_copy = amount + малый шум (corr ~ 0.99)
    df_train["amount_copy"] = df_train["amount"] + rng.normal(0, 5, n_train)
    df_train["target"] = fraud_train

    # --- OOT ------------------------------------------------------------------
    fraud_oot = rng.integers(0, 2, n_oot)

    df_oot = pd.DataFrame(
        {
            "amount":      rng.normal(500, 200, n_oot) + fraud_oot * 300,
            "velocity_1h": rng.exponential(2, n_oot) + fraud_oot * 5,
            "card_age":    rng.uniform(0, 10, n_oot) - fraud_oot * 2,
            "weak_feat":   rng.normal(0, 1, n_oot) + fraud_oot * 0.05,
            "amount_copy": None,
            "noise_1":     rng.normal(0, 1, n_oot),
            "noise_2":     rng.uniform(0, 1, n_oot),
            # Сильный дрейф: среднее сдвинуто на 10σ — PSI >> 0.2
            "drifting":    rng.normal(10, 1, n_oot) + fraud_oot * 3,
            "dist_center": rng.exponential(10, n_oot) + fraud_oot * 8,
            "hour_sin":    np.sin(rng.uniform(0, 2 * np.pi, n_oot)) + fraud_oot * 0.3,
        }
    )
    df_oot["amount_copy"] = df_oot["amount"] + rng.normal(0, 5, n_oot)

    feature_cols = [
        "amount", "velocity_1h", "card_age", "weak_feat", "amount_copy",
        "noise_1", "noise_2", "drifting", "dist_center", "hour_sin",
    ]

    # --- Fit ------------------------------------------------------------------
    selector = FeatureSelector(
        iv_threshold=0.05,
        correlation_threshold=0.8,
        psi_threshold=0.2,
    )
    selector.fit(df_train, feature_cols, "target", df_oot=df_oot)

    # --- Report ---------------------------------------------------------------
    report = selector.get_selection_report()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n=== Selection Report ===")
    print(report.to_string(index=False))

    print(f"\nSelected features ({len(selector.selected_features_)}): {selector.selected_features_}")
    print(f"Dropped features  ({len(selector.dropped_features_)}): {selector.dropped_features_}")

    # --- Transform sanity check -----------------------------------------------
    df_selected = selector.transform(df_train)
    assert list(df_selected.columns) == selector.selected_features_
    assert len(df_selected) == n_train
    print("\ntransform() OK — shape:", df_selected.shape)
