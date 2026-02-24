"""Smoke test: 1000 транзакций, 100 пользователей."""

import logging
import sys

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    stream=sys.stdout,
)

# -----------------------------------------------------------------------
# 1. Генерация тестовых данных
# -----------------------------------------------------------------------
np.random.seed(42)

N_TX = 1000
N_USERS = 100
FRAUD_RATE = 0.08

user_ids = np.random.randint(1, N_USERS + 1, size=N_TX)
amounts = np.abs(np.random.lognormal(mean=4.5, sigma=1.2, size=N_TX))
start_date = pd.Timestamp("2024-01-01")
timestamps = [start_date + pd.Timedelta(hours=int(h)) for h in np.sort(np.random.randint(0, 365 * 24, size=N_TX))]
merchant_ids = np.random.randint(1, 50, size=N_TX)

# Фрод: высокая сумма + небольшое число уникальных мерчантов
fraud_prob = 0.02 + 0.15 * (amounts > np.percentile(amounts, 85)).astype(float)
fraud_labels = (np.random.rand(N_TX) < fraud_prob).astype(int)

df = pd.DataFrame({
    "user_id": user_ids,
    "amount": amounts.round(2),
    "timestamp": timestamps,
    "merchant_id": merchant_ids,
    "is_fraud": fraud_labels,
})

print(f"\n{'='*60}")
print(f"Тестовый датасет: {len(df)} транзакций, {df['user_id'].nunique()} пользователей")
print(f"Фрод-rate: {df['is_fraud'].mean():.2%}")
print(f"{'='*60}\n")

# -----------------------------------------------------------------------
# 2. fit_transform
# -----------------------------------------------------------------------
from src.pipeline import AntiFraudPipeline  # noqa: E402

pipeline = AntiFraudPipeline(
    time_windows=[7, 30, 90],
    iv_threshold=0.02,
    correlation_threshold=0.9,
    max_n_bins=8,
)

X_transformed = pipeline.fit_transform(
    df,
    target_col="is_fraud",
    user_col="user_id",
    amount_col="amount",
    timestamp_col="timestamp",
)

print(f"\nРезультат fit_transform: {X_transformed.shape[0]} строк × {X_transformed.shape[1]} фич")
print(f"Фичи: {list(X_transformed.columns)}\n")

# -----------------------------------------------------------------------
# 3. get_feature_importance().head(10)
# -----------------------------------------------------------------------
print("=" * 60)
print("TOP-10 фич по IV:")
print("=" * 60)
importance = pipeline.get_feature_importance()
print(importance.head(10).to_string(index=False))

# -----------------------------------------------------------------------
# 4. OOT-метрики
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("OOT Метрики:")
print("=" * 60)
metrics = pipeline.get_oot_metrics()
for k, v in metrics.items():
    print(f"  {k:20s}: {v}")

# -----------------------------------------------------------------------
# 5. Gain Chart
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("Gain Chart (OOT):")
print("=" * 60)
gain = pipeline.get_gain_chart()
if not gain.empty:
    print(gain.to_string(index=False))

# -----------------------------------------------------------------------
# 6. Save / Load
# -----------------------------------------------------------------------
SAVE_PATH = "/tmp/antifraud_pipeline.pkl"
pipeline.save(SAVE_PATH)

loaded = AntiFraudPipeline.load(SAVE_PATH)
print(f"\nPipeline сохранён и загружен из {SAVE_PATH}")
print(f"Loaded pipeline is_fitted: {loaded._is_fitted}")

# Verify loaded pipeline works
X2 = loaded.transform(df.head(20))
print(f"transform на 20 строках через загруженный pipeline: shape={X2.shape}")

# -----------------------------------------------------------------------
# 7. SHAP (опционально)
# -----------------------------------------------------------------------
try:
    import shap  # noqa: F401
    print("\n" + "=" * 60)
    print("SHAP values (первые 5 транзакций):")
    print("=" * 60)
    shap_df = pipeline.explain(df.head(50), sample_size=50)
    print(shap_df.head(5).round(4).to_string())
except ImportError:
    print("\n[INFO] shap не установлен — SHAP-шаг пропущен. pip install shap")

print("\nSmoke test завершён успешно.")
