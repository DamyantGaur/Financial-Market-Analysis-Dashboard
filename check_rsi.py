
import pandas as pd
import numpy as np

try:
    import ta
    print("✅ 'ta' library is installed.")
except ImportError:
    print("❌ 'ta' library is NOT installed.")

# Test manual implementation
def compute_rsi_manual(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    # Wilder's smoothing
    for i in range(period, len(avg_gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period
        
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Generate dummy data
print("\nTesting RSI on dummy data...")
price = pd.Series(np.random.randn(100).cumsum() + 100)
rsi = compute_rsi_manual(price)
print(f"RSI tail:\n{rsi.tail()}")

if rsi.isnull().all():
    print("❌ RSI calculation returned all NaNs!")
else:
    print("✅ RSI calculation returned values.")
