"""Anti-fraud feature pipeline: generation → binning → selection → model."""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    roc_auc_score,
)
from scipy import stats

from src.binning import FeatureBinner
from src.feature_engineering import FeatureGenerator
from src.feature_selection import FeatureSelector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper metrics
# ---------------------------------------------------------------------------

def _precision_at_top_n(y_true: np.ndarray, y_score: np.ndarray, top_frac: float) -> float:
    """Precision среди top_frac*100% самых рискованных транзакций."""
    n = max(1, int(len(y_true) * top_frac))
    top_idx = np.argsort(y_score)[::-1][:n]
    return float(np.mean(y_true[top_idx]))


def _ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS-статистика: максимальная разница между TPR и FPR."""
    fraud_scores = y_score[y_true == 1]
    legit_scores = y_score[y_true == 0]
    if len(fraud_scores) == 0 or len(legit_scores) == 0:
        return 0.0
    ks, _ = stats.ks_2samp(fraud_scores, legit_scores)
    return float(ks)


def _gain_chart(y_true: np.ndarray, y_score: np.ndarray, n_buckets: int = 10) -> pd.DataFrame:
    """Строит Gain Chart — сколько фрода поймали в топ-X% транзакций."""
    df = pd.DataFrame({"score": y_score, "label": y_true}).sort_values(
        "score", ascending=False
    ).reset_index(drop=True)
    total_fraud = y_true.sum()
    rows = []
    for i in range(1, n_buckets + 1):
        cutoff = int(len(df) * i / n_buckets)
        captured = df["label"].iloc[:cutoff].sum()
        rows.append({
            "top_pct": i * 10,
            "captured_fraud": int(captured),
            "gain": captured / total_fraud if total_fraud > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class AntiFraudPipeline:
    """Полный pipeline для подготовки фич и оценки антифрод модели.

    Этапы:
    1. Генерация агрегатных фич (FeatureGenerator)
    2. Биннинг числовых фич с WoE/IV (FeatureBinner)
    3. Отбор по IV и корреляции (FeatureSelector)
    4. Калиброванная модель (GBT + CalibratedClassifierCV)

    Оценка на OOT-срезе:
    - PR-AUC
    - Precision@Top-1% и Top-5%
    - KS-статистика
    - Gain Chart
    - SHAP-значения (опционально)
    """

    def __init__(
        self,
        time_windows: list = None,
        iv_threshold: float = 0.1,
        correlation_threshold: float = 0.8,
        max_n_bins: int = 10,
    ):
        self.time_windows = time_windows or [7, 30, 90]
        self.iv_threshold = iv_threshold
        self.correlation_threshold = correlation_threshold
        self.max_n_bins = max_n_bins

        self._generator = FeatureGenerator(time_windows=self.time_windows)
        self._binner = FeatureBinner(
            iv_threshold=0.02,  # мягкий порог на биннере; жёсткий — в selector
            max_n_bins=self.max_n_bins,
        )
        self._selector = FeatureSelector(
            iv_threshold=self.iv_threshold,
            correlation_threshold=self.correlation_threshold,
        )

        base_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self._model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)

        # populated after fit
        self._feature_importance: Optional[pd.DataFrame] = None
        self._oot_metrics: Optional[dict] = None
        self._gain_chart: Optional[pd.DataFrame] = None
        self._shap_values: Optional[np.ndarray] = None
        self._shap_feature_names: Optional[list[str]] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        user_col: str = "user_id",
        amount_col: str = "amount",
        timestamp_col: str = "timestamp",
        additional_features: list = None,
        oot_frac: float = 0.2,
    ) -> "AntiFraudPipeline":
        """Обучает весь pipeline.

        Parameters
        ----------
        df:
            Сырые транзакционные данные.
        target_col:
            Бинарная целевая переменная (1 = фрод).
        oot_frac:
            Доля последних (по времени) транзакций, отводимая под OOT-тест.
        additional_features:
            Дополнительные числовые колонки, включаемые напрямую (без генерации).
        """
        self._target_col = target_col
        self._user_col = user_col
        self._amount_col = amount_col
        self._timestamp_col = timestamp_col
        self._additional_features = additional_features or []

        df = df.sort_values(timestamp_col).reset_index(drop=True)

        # --- Time-based split -------------------------------------------
        split_idx = int(len(df) * (1 - oot_frac))
        train_df = df.iloc[:split_idx].copy()
        oot_df = df.iloc[split_idx:].copy()
        logger.info(
            "Time-based split: train=%d, OOT=%d rows", len(train_df), len(oot_df)
        )

        # --- Step 1: Feature generation (train) -------------------------
        logger.info("Step 1: FeatureGenerator.fit_transform на train...")
        agg_train = self._generator.fit_transform(
            train_df,
            user_col=user_col,
            amount_col=amount_col,
            timestamp_col=timestamp_col,
        )

        # --- Build combined train feature matrix ------------------------
        X_train, y_train = self._build_feature_matrix(
            train_df, agg_train, target_col
        )

        # --- Step 2: Binning -------------------------------------------
        logger.info("Step 2: FeatureBinner.fit...")
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train_for_binner = X_train.assign(_y=y_train.values)
        self._binner.fit(X_train_for_binner, num_cols, "_y")
        X_train_woe = self._binner.transform(X_train)

        # --- Step 3: Feature selection ---------------------------------
        logger.info("Step 3: FeatureSelector.fit...")
        iv_table = self._binner.get_iv_table()
        self._selector.fit(X_train_woe, iv_table)

        X_train_sel = self._selector.transform(X_train_woe)
        if X_train_sel.empty or X_train_sel.shape[1] == 0:
            raise RuntimeError(
                "После отбора фич не осталось ни одной. "
                "Снизьте iv_threshold или correlation_threshold."
            )

        self._selected_cols = self._selector.get_selected_features()
        logger.info("Отобрано фич: %d — %s", len(self._selected_cols), self._selected_cols)

        # --- Step 4: Train calibrated model ----------------------------
        logger.info("Step 4: обучение CalibratedClassifierCV...")
        self._model.fit(X_train_sel.values, y_train.values)

        # --- Build feature importance table ----------------------------
        self._feature_importance = self._build_importance_table(iv_table)

        # Mark as fitted before OOT evaluation so predict_proba works
        self._is_fitted = True

        # --- OOT evaluation --------------------------------------------
        logger.info("OOT evaluation...")
        self._evaluate_oot(oot_df)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применяет все трансформации (без обучения)."""
        self._check_fitted()
        agg = self._generator.transform(df)
        X, _ = self._build_feature_matrix(df, agg, target_col=None)
        X_woe = self._binner.transform(X)
        X_sel = self._selector.transform(X_woe)
        return X_sel

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """fit + transform на тех же данных."""
        self.fit(df, **kwargs)
        return self.transform(df)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Возвращает P(fraud) для каждой транзакции."""
        X = self.transform(df)
        return self._model.predict_proba(X.values)[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        """Итоговая таблица важности фич: feature | IV | predictive_power | selected."""
        self._check_fitted()
        return self._feature_importance.copy()

    def get_oot_metrics(self) -> dict:
        """Метрики на OOT-срезе: pr_auc, precision@1%, precision@5%, ks, roc_auc."""
        self._check_fitted()
        return dict(self._oot_metrics)

    def get_gain_chart(self) -> pd.DataFrame:
        """Gain Chart по OOT-данным."""
        self._check_fitted()
        return self._gain_chart.copy()

    def explain(
        self,
        df: pd.DataFrame,
        sample_size: int = 200,
    ) -> pd.DataFrame:
        """Возвращает SHAP-значения для выборки транзакций.

        Requires: pip install shap

        Returns DataFrame (n_samples × n_features) с SHAP-значениями.
        """
        try:
            import shap  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("Установите shap: pip install shap") from e

        self._check_fitted()
        X = self.transform(df)
        if len(X) > sample_size:
            X = X.sample(sample_size, random_state=42)

        # Extract underlying base estimator for TreeExplainer
        base_clf = self._model.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(base_clf)
        shap_values = explainer.shap_values(X.values)
        return pd.DataFrame(shap_values, columns=X.columns, index=X.index)

    def save(self, path: str) -> None:
        """Сохраняет pipeline в pickle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Pipeline сохранён: %s", path)

    @classmethod
    def load(cls, path: str) -> "AntiFraudPipeline":
        """Загружает pipeline из pickle."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Загруженный объект не является {cls.__name__}")
        logger.info("Pipeline загружен: %s", path)
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_feature_matrix(
        self,
        raw_df: pd.DataFrame,
        agg_df: pd.DataFrame,
        target_col: Optional[str],
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """Собирает матрицу фич из агрегатов + дополнительных колонок."""
        frames = [agg_df]

        if self._additional_features:
            available = [
                c for c in self._additional_features if c in raw_df.columns
            ]
            if available:
                frames.append(raw_df[available].reset_index(drop=True))

        X = pd.concat(frames, axis=1)

        y = None
        if target_col and target_col in raw_df.columns:
            y = raw_df[target_col].reset_index(drop=True).rename("target")

        return X, y

    def _build_importance_table(self, iv_table: pd.DataFrame) -> pd.DataFrame:
        selected_set = set(self._selector.get_selected_features())
        rows = []
        for _, row in iv_table.iterrows():
            feat = row["feature"]
            rows.append(
                {
                    "feature": feat,
                    "IV": row["IV"],
                    "predictive_power": row["predictive_power"],
                    "selected": feat in selected_set,
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("IV", ascending=False)
            .reset_index(drop=True)
        )

    def _evaluate_oot(self, oot_df: pd.DataFrame) -> None:
        """Оценка модели на OOT-данных с расчётом всех метрик."""
        if self._target_col not in oot_df.columns:
            logger.warning("Целевая переменная отсутствует в OOT — пропуск оценки.")
            self._oot_metrics = {}
            self._gain_chart = pd.DataFrame()
            return

        y_oot = oot_df[self._target_col].values
        if y_oot.sum() == 0:
            logger.warning("OOT не содержит фрод-транзакций — пропуск оценки.")
            self._oot_metrics = {}
            self._gain_chart = pd.DataFrame()
            return

        try:
            scores = self.predict_proba(oot_df)
        except Exception as exc:
            logger.warning("Ошибка predict_proba на OOT: %s", exc)
            self._oot_metrics = {}
            self._gain_chart = pd.DataFrame()
            return

        pr_auc = average_precision_score(y_oot, scores)
        roc_auc = roc_auc_score(y_oot, scores)
        ks = _ks_statistic(y_oot, scores)
        p1 = _precision_at_top_n(y_oot, scores, 0.01)
        p5 = _precision_at_top_n(y_oot, scores, 0.05)

        self._oot_metrics = {
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "ks": round(ks, 4),
            "precision@1%": round(p1, 4),
            "precision@5%": round(p5, 4),
        }
        self._gain_chart = _gain_chart(y_oot, scores)

        logger.info("OOT metrics: %s", self._oot_metrics)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Pipeline не обучен — вызовите fit().")
