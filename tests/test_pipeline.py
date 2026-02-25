"""Tests for all anti-fraud pipeline modules.

Run with:
    pytest tests/test_pipeline.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.binning import FeatureBinner
from src.feature_engineering import FeatureGenerator
from src.feature_selection import FeatureSelector
from src.pipeline import AntiFraudPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_transactions():
    """Создаёт тестовый DataFrame с транзакциями.

    1 000 строк, ~5 % фрода.  Мошеннические транзакции имеют намеренно
    завышенную сумму — это даёт ненулевой IV для WoE-биннинга.
    """
    rng = np.random.default_rng(42)
    n = 1_000

    is_fraud = rng.binomial(1, 0.05, n)  # 5 % fraud rate

    # Frauds tend to have higher amounts → provides IV signal
    amount = np.where(
        is_fraud.astype(bool),
        rng.exponential(scale=500, size=n),
        rng.exponential(scale=100, size=n),
    )

    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")

    return pd.DataFrame(
        {
            "transaction_id": range(n),
            "card_id": rng.integers(1, 50, n),
            "merchant_id": rng.integers(1, 20, n),
            "amount": amount,
            "timestamp": timestamps,
            "is_fraud": is_fraud,
        }
    )


# ---------------------------------------------------------------------------
# TestFeatureBinner
# ---------------------------------------------------------------------------


class TestFeatureBinner:
    """Тесты для src.binning.FeatureBinner."""

    def _prepare_binner_input(self, df: pd.DataFrame):
        """Вспомогательный метод: генерирует признаки и возвращает (train_df, feature_cols)."""
        gen = FeatureGenerator()
        features = gen.fit_transform(df)
        feature_cols = list(features.columns)
        train_df = features.copy()
        train_df["is_fraud"] = df["is_fraud"].values
        return train_df, feature_cols

    def test_fit_transform(self, sample_transactions):
        """Тест что fit/transform работают без ошибок."""
        train_df, feature_cols = self._prepare_binner_input(sample_transactions)

        binner = FeatureBinner(iv_threshold=0.02)
        binner.fit(train_df, feature_cols=feature_cols, target_col="is_fraud")
        result = binner.transform(train_df)

        assert isinstance(result, pd.DataFrame), "transform должен вернуть DataFrame"
        assert len(result) == len(sample_transactions), (
            "Количество строк не должно измениться"
        )
        assert result.shape[1] > 0, "Должен остаться хотя бы один признак"

    def test_iv_threshold(self, sample_transactions):
        """Тест что фичи с низким IV отфильтровываются."""
        df = sample_transactions.copy()
        # Добавляем «бесполезный» шумовой признак
        rng = np.random.default_rng(0)
        df["random_noise"] = rng.normal(0, 1, len(df))

        gen = FeatureGenerator()
        features = gen.fit_transform(df)
        features["random_noise"] = df["random_noise"].values

        feature_cols = list(features.columns)
        train_df = features.copy()
        train_df["is_fraud"] = df["is_fraud"].values

        binner = FeatureBinner(iv_threshold=0.02)
        binner.fit(train_df, feature_cols=feature_cols, target_col="is_fraud")

        iv_table = binner.get_iv_table()
        noise_rows = iv_table[iv_table["feature"] == "random_noise"]

        if not noise_rows.empty:
            noise_iv = float(noise_rows["IV"].iloc[0])
            if noise_iv < 0.02:
                assert "random_noise" not in binner.binners_, (
                    f"random_noise (IV={noise_iv:.4f}) должен быть исключён"
                )

    def test_iv_table_format(self, sample_transactions):
        """Тест формата таблицы IV."""
        train_df, feature_cols = self._prepare_binner_input(sample_transactions)

        binner = FeatureBinner(iv_threshold=0.02)
        binner.fit(train_df, feature_cols=feature_cols, target_col="is_fraud")
        iv_table = binner.get_iv_table()

        # Тип
        assert isinstance(iv_table, pd.DataFrame)

        # Обязательные колонки
        required_cols = {"feature", "IV", "n_bins", "predictive_power"}
        assert required_cols.issubset(set(iv_table.columns)), (
            f"Не хватает колонок: {required_cols - set(iv_table.columns)}"
        )

        # Допустимые метки предсказательной силы
        valid_labels = {"useless", "weak", "medium", "strong"}
        actual_labels = set(iv_table["predictive_power"].unique())
        assert actual_labels.issubset(valid_labels), (
            f"Недопустимые метки: {actual_labels - valid_labels}"
        )

        # IV должен быть отсортирован по убыванию
        iv_values = list(iv_table["IV"])
        assert iv_values == sorted(iv_values, reverse=True), (
            "iv_table должен быть отсортирован по IV убывающе"
        )


# ---------------------------------------------------------------------------
# TestFeatureGenerator
# ---------------------------------------------------------------------------


class TestFeatureGenerator:
    """Тесты для src.feature_engineering.FeatureGenerator."""

    def test_feature_count(self, sample_transactions):
        """Тест что генерируется ожидаемое количество фич."""
        gen = FeatureGenerator()
        features = gen.fit_transform(sample_transactions)

        expected = len(FeatureGenerator.EXPECTED_BASE_FEATURES)
        assert features.shape[1] == expected, (
            f"Ожидалось {expected} признаков, получено {features.shape[1]}: "
            f"{list(features.columns)}"
        )

    def test_no_nan_in_output(self, sample_transactions):
        """Тест что нет NaN в результате."""
        gen = FeatureGenerator()
        features = gen.fit_transform(sample_transactions)

        nan_counts = features.isna().sum()
        nan_features = nan_counts[nan_counts > 0]

        assert nan_features.empty, (
            f"Найдены NaN в признаках: {nan_features.to_dict()}"
        )


# ---------------------------------------------------------------------------
# TestFeatureSelector
# ---------------------------------------------------------------------------


class TestFeatureSelector:
    """Тесты для src.feature_selection.FeatureSelector."""

    def test_correlation_removal(self):
        """Тест удаления коррелированных фич."""
        rng = np.random.default_rng(0)
        n = 300

        base = rng.normal(0, 1, n)
        df = pd.DataFrame(
            {
                "feature_a": base,
                # ~1.0 корреляция с feature_a → должна быть удалена
                "feature_b": base + rng.normal(0, 0.01, n),
                # независимая → должна остаться
                "feature_c": rng.normal(0, 1, n),
            }
        )

        selector = FeatureSelector(correlation_threshold=0.95)
        selector.fit(df)
        result = selector.transform(df)

        assert "feature_b" not in result.columns, (
            "feature_b сильно коррелирована с feature_a и должна быть удалена"
        )
        assert "feature_c" in result.columns, (
            "feature_c независима и должна остаться"
        )
        # Хотя бы feature_a и feature_c должны присутствовать
        assert result.shape[1] >= 2

    def test_selection_report(self):
        """Тест что отчёт содержит все фичи."""
        rng = np.random.default_rng(1)
        n = 150
        cols = [f"f{i}" for i in range(5)]
        df = pd.DataFrame(rng.normal(0, 1, (n, len(cols))), columns=cols)

        selector = FeatureSelector(correlation_threshold=0.95)
        selector.fit(df)
        report = selector.get_selection_report()

        assert isinstance(report, pd.DataFrame)

        # Обязательные колонки
        required_cols = {"feature", "max_correlation", "selected"}
        assert required_cols.issubset(set(report.columns)), (
            f"Не хватает колонок: {required_cols - set(report.columns)}"
        )

        # Все исходные признаки должны быть в отчёте
        assert set(report["feature"]) == set(df.columns), (
            "Отчёт должен содержать все исходные признаки"
        )

        # selected должен быть boolean
        assert report["selected"].dtype == bool, (
            f"Колонка 'selected' должна быть bool, получено: {report['selected'].dtype}"
        )


# ---------------------------------------------------------------------------
# TestAntiFraudPipeline
# ---------------------------------------------------------------------------


class TestAntiFraudPipeline:
    """Интеграционные тесты для src.pipeline.AntiFraudPipeline."""

    def test_end_to_end(self, sample_transactions):
        """Интеграционный тест всего pipeline."""
        pipeline = AntiFraudPipeline(iv_threshold=0.02)
        result = pipeline.fit_transform(sample_transactions, target_col="is_fraud")

        assert isinstance(result, pd.DataFrame), "Результат должен быть DataFrame"
        assert len(result) == len(sample_transactions), (
            "Количество строк должно совпадать с входными данными"
        )
        assert result.shape[1] > 0, "Должен остаться хотя бы один признак"
        assert not result.isna().any().any(), (
            "Pipeline не должен возвращать NaN значения"
        )

    def test_save_load(self, sample_transactions, tmp_path):
        """Тест сохранения и загрузки pipeline."""
        pipeline = AntiFraudPipeline(iv_threshold=0.02)
        pipeline.fit(sample_transactions, target_col="is_fraud")

        save_path = tmp_path / "pipeline.pkl"
        pipeline.save(save_path)

        assert save_path.exists(), "Файл pipeline должен быть создан"

        loaded = AntiFraudPipeline.load(save_path)
        assert isinstance(loaded, AntiFraudPipeline), (
            "Загруженный объект должен быть экземпляром AntiFraudPipeline"
        )

        # Результаты оригинала и загруженного pipeline должны совпадать
        result_original = pipeline.transform(sample_transactions)
        result_loaded = loaded.transform(sample_transactions)

        pd.testing.assert_frame_equal(
            result_original,
            result_loaded,
            check_names=True,
            obj="pipeline.transform()",
        )
