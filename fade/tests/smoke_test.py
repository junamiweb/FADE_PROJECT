import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

from fade.core import dl_model
from fade.config import lean_config

def run_smoke_test():
    print("Starting FADE v0.3 Smoke Test...")
    # Use the news_dl set we just added
    config = lean_config("news_dl")
    
    # Create dummy OHLCV data
    data_path = "test_btc.csv"
    news_path = "test_news.csv"
    
    print("Creating dummy test data...")
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "open": np.random.randn(100).cumsum() + 1000,
        "high": np.random.randn(100).cumsum() + 1010,
        "low": np.random.randn(100).cumsum() + 990,
        "close": np.random.randn(100).cumsum() + 1000,
        "volume": np.random.randint(100, 1000, 100)
    }, index=dates)
    df.index.name = "timestamp"
    df.to_csv(data_path)
    
    # Create dummy news data. Start well before the price window so the
    # 30-day news_vol_z rolling window is already warmed up once price bars
    # begin (otherwise every bar gets dropped by dropna in build_sequential_dataset).
    news_start = pd.Timestamp("2023-01-01") - pd.Timedelta(days=35)
    news_df = pd.DataFrame({
        "date": pd.date_range(news_start, periods=40, freq="D").strftime("%Y-%m-%d"),
        "tone": np.random.uniform(-1, 1, 40),
        "volume": np.random.randint(10, 100, 40)
    })
    news_df.to_csv(news_path, index=False)
    
    try:
        print(f"DL Backend: {dl_model.dl_backend_name()}")
        assert dl_model.dl_backend_available(), "PyTorch backend not found"
        
        print("1. Testing Dataset Builder (with news integration)...")
        X, y, fwd, meta = dl_model.build_sequential_dataset(
            data_path, config, news_csv=news_path, 
            feature_cols=config.atom_columns
        )
        print(f"   - X shape: {X.shape}")
        assert X.shape[0] > 0, "Dataset should not be empty"
        assert X.shape[2] == len(config.atom_columns), "Feature count mismatch"
        
        print("2. Testing Training Loop (1 epoch)...")
        # Low holdout for quick test
        report = dl_model.train_and_evaluate(
            data_path, config, news_csv=news_path, 
            epochs=2, holdout_frac=0.2,
            feature_cols=config.atom_columns
        )
        print(f"   - Status: {report['status']}")
        print(f"   - Last Direction: {report.get('last_direction')}")
        assert report['status'] in ['ok', 'too_short'], f"Unexpected status: {report['status']}"

        print("3. Testing Forecast function...")
        f_report = dl_model.forecast_latest(
            data_path, config, news_csv=news_path, 
            epochs=1, feature_cols=config.atom_columns
        )
        print(f"   - Forecast Direction: {f_report.get('last_direction')}")
        
        print("\nSUCCESS: All components of the new DL+News pipeline are functional!")
        
    finally:
        if Path(data_path).exists(): Path(data_path).unlink()
        if Path(news_path).exists(): Path(news_path).unlink()

if __name__ == "__main__":
    run_smoke_test()
