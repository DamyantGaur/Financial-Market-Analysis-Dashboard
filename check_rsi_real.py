
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath("src"))

try:
    from technical_analysis import compute_rsi
    print("✅ Imported compute_rsi from technical_analysis.")
except ImportError as e:
    print(f"❌ Failed to import compute_rsi: {e}")
    sys.exit(1)

# Generate dummy data
print("\nTesting compute_rsi on dummy data...")
price = pd.Series(np.random.randn(100).cumsum() + 100)
df = pd.DataFrame({"Close": price})

try:
    result = compute_rsi(df)
    rsi = result["RSI"]
    print(f"RSI tail:\n{rsi.tail()}")

    if rsi.isnull().all():
        print("❌ RSI calculation returned all NaNs!")
    else:
        print("✅ RSI calculation returned values.")
        
except Exception as e:
    print(f"❌ RSI calculation failed: {e}")
