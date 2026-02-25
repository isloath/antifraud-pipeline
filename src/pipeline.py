"""End-to-end AntiFraud pipeline: FeatureGenerator → FeatureSelector → FeatureBinner."""

import logging
import pickle
from pathlib import Path

import pandas as pd

from src.binning import FeatureBinner
from src.feature_engineering import FeatureGenerator
from src.feature_selection import FeatureSelector

logger = logging.getLogger(__name__)


class AntiFraudPipeline:
    """Объединяет три шага в единый обучаемый/применяемый объект.

    Steps
    -----
    1. **FeatureGenerator** — строит 13 числовых признаков из сырых транзакций.
    2. **FeatureSelector** — удаляет высококоррелированные признаки.
    3. **FeatureBinner** — биннинг WoE/IV с отсевом слабых признаков.

    Parameters
    ----------
    iv_threshold : float
        Минимальный IV для сохранения признака (по умолчанию 0.02).
    max_n_bins : int
        Максимальное число бинов в OptimalBinning (по умолчанию 10).
    correlation_threshold : float
        Порог корреляции для FeatureSelector (по умолчанию 0.95).
    timestamp_col, card_col, merchant_col, amount_col : str
        Названия входных колонок исходного DataFrame.
    """

    def __init__(
        self,
        iv_threshold: float = 0.02,
        max_n_bins: int = 10,
        correlation_threshold: float = 0.95,
        timestamp_col: str = "timestamp",
        card_col: str = "card_id",
        merchant_col: str = "merchant_id",
        amount_col: str = "amount",
    ):
        self.iv_threshold = iv_threshold
        self.max_n_bins = max_n_bins
        self.correlation_threshold = correlation_threshold
        self.timestamp_col = timestamp_col
        self.card_col = card_col
        self.merchant_col = merchant_col
        self.amount_col = amount_col

        self.generator_ = FeatureGenerator(
            timestamp_col=timestamp_col,
            card_col=card_col,
            merchant_col=merchant_col,
            amount_col=amount_col,
        )
        self.selector_ = FeatureSelector(
            correlation_threshold=correlation_threshold
        )
        self.binner_ = FeatureBinner(
            iv_threshold=iv_threshold, max_n_bins=max_n_bins
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, target_col: str = "is_fraud") -> "AntiFraudPipeline":
        """Обучает весь pipeline на тренировочных данных.

        Parameters
        ----------
        df : pd.DataFrame
            Сырые транзакции со всеми необходимыми колонками + target_col.
        target_col : str
            Название колонки с целевой переменной (0/1).
        """
        # Step 1: generate features
        features = self.generator_.fit_transform(df)

        # Step 2: select features
        self.selector_.fit(features)
        selected = self.selector_.transform(features)

        # Step 3: fit binner (needs target alongside features)
        feature_cols = list(selected.columns)
        train_df = selected.copy()
        train_df[target_col] = df[target_col].values
        self.binner_.fit(train_df, feature_cols=feature_cols, target_col=target_col)

        logger.info(
            "AntiFraudPipeline обучен: %d признаков после биннинга",
            len(self.binner_.binners_),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применяет обученный pipeline к новым данным.

        Returns
        -------
        pd.DataFrame
            WoE-трансформированные признаки (только те, что прошли IV-фильтр).
        """
        features = self.generator_.transform(df)
        selected = self.selector_.transform(features)
        return self.binner_.transform(selected)

    def fit_transform(
        self, df: pd.DataFrame, target_col: str = "is_fraud"
    ) -> pd.DataFrame:
        """Обучает pipeline и сразу возвращает трансформированные данные."""
        self.fit(df, target_col=target_col)
        return self.transform(df)

    def save(self, path) -> None:
        """Сохраняет pipeline через pickle.

        Parameters
        ----------
        path : str | Path
            Путь к файлу (родительские директории создаются автоматически).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("AntiFraudPipeline сохранён: %s", path)

    @classmethod
    def load(cls, path) -> "AntiFraudPipeline":
        """Загружает ранее сохранённый pipeline.

        Parameters
        ----------
        path : str | Path
            Путь к pickle-файлу.

        Returns
        -------
        AntiFraudPipeline
        """
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Ожидался AntiFraudPipeline, получен {type(obj).__name__}"
            )
        logger.info("AntiFraudPipeline загружен из: %s", path)
        return obj
